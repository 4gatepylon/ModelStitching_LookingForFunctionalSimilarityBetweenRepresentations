import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from resnet import RESNETS_FOLDER

from download import (
    cifar_models_from_imagenet_models
)

from utils import fix_seed, evaluate, adjust_learning_rate

from typing import List

from torch.cuda.amp import GradScaler, autocast
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]
label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]


def get_loaders():
    num_of_points = 50000
    split = [int(num_of_points * args.fraction), int(num_of_points * (1 - args.fraction))]

    dataset_class = torchvision.datasets.CIFAR10 if (args.dataset == 'cifar10') else torchvision.datasets.CIFAR100
    if not os.path.exists(f'tmp/finetune_{args.dataset}_{args.fraction}_train_data.beton'):
        train_data = dataset_class(
            '../Data', train=True, download=True
        )
        train_data = torch.utils.data.random_split(train_data, split)[0]
        train_writer = DatasetWriter(f'tmp/finetune_{args.dataset}_{args.fraction}_train_data.beton', {
            'image': RGBImageField(),
            'label': IntField()
        })
        train_writer.from_indexed_dataset(train_data)

    image_pipeline_train: List[Operation] = [
        SimpleRGBImageDecoder(),
        RandomHorizontalFlip(),
        RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
        Cutout(4, tuple(map(int, CIFAR_MEAN))),
        ToTensor(),
        ToDevice('cuda:0', non_blocking=True),
        ToTorchImage(),
        Convert(torch.float16),
        torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]
    train_loader = Loader(f'./tmp/finetune_{args.dataset}_{args.fraction}_train_data.beton',
                          batch_size=args.bsz,
                          num_workers=args.num_workers,
                          order=OrderOption.RANDOM,
                          os_cache=True,
                          drop_last=True,
                          pipelines={
                              'image': image_pipeline_train,
                              'label': label_pipeline
                          })

    if not os.path.exists(f'tmp/{args.dataset}_test_data.beton'):
        test_data = dataset_class(
            '../Data', train=False, download=True
        )

        test_writer = DatasetWriter(f'tmp/{args.dataset}_test_data.beton', {
            'image': RGBImageField(),
            'label': IntField()
        })
        test_writer.from_indexed_dataset(test_data)

    image_pipeline_test: List[Operation] = [
        SimpleRGBImageDecoder(),
        ToTensor(),
        ToDevice('cuda:0', non_blocking=True),
        ToTorchImage(),
        Convert(torch.float16),
        torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]

    test_loader = Loader(f'./tmp/{args.dataset}_test_data.beton',
                         batch_size=2048,
                         num_workers=args.num_workers,
                         order=OrderOption.SEQUENTIAL,
                         os_cache=True,
                         drop_last=False,
                         pipelines={
                             'image': image_pipeline_test,
                             'label': label_pipeline
                         })
    return train_loader, test_loader


def train_loop(model, train_loader, test_loader, parameters=None, epochs=None):
    # None signifies do all parameters (we might finetune single layers an that will speed up training)
    if parameters is None:
       parameters = list(model.parameters())

    optimizer = torch.optim.SGD(
        params=parameters,
        momentum=0.9,
        lr=args.lr * args.bsz / 256,
        weight_decay=args.wd
    )

    scaler = GradScaler()

    start = time.time()
    epochs = args.epochs if epochs is None else epochs
    for e in range(1, epochs + 1):
        model.train()
        # epoch
        for it, (inputs, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):

            # adjust
            adjust_learning_rate(epochs=epochs,
                                 warmup_epochs=args.warmup,
                                 base_lr=args.lr * args.bsz / 256,
                                 optimizer=optimizer,
                                 loader=train_loader,
                                 step=it)
            # zero grad
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                h = inputs
                h = model(h)
                loss = F.cross_entropy(h, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f'epoch: {e} | time: {time.time() - start:.3f}')

    eval_acc = evaluate(model, test_loader)

    print(f'final_eval_acc: {eval_acc:.3f}')

    # torch.save(model[0].state_dict(), 'cifar_resnet18_supervised.pth') # NOTE I removed this since I save outside
    return eval_acc, model

# We can shoot to do stitching with
# resnet18 (basic block)
# resnet34 (basic block)

# NOTE these are the experiments we have planned
# - resnet18 block2block (including within block sets: mistake I did before)
# - random resnet18 block2block with resnet18
# - resnet18 block2block w/ resnet34
# - random resnet18 block2block w/ resnet34

# NOTE right now this just finetunes the pretrained models so we can run our experiments tomorrow
def main_pretrain(args):
    fix_seed(args.seed)

    train_loader, test_loader = get_loaders()
    models = cifar_models_from_imagenet_models()

    # Decide epochs such that they will make our model training slightly better
    desired_keys = ["resnet18", "resnet34", "resnet50", "wide_resnet50_2"]
    desired_epochs = [40, 60, 80, 100]
    desired_epochs = {desired_keys[i] : desired_epochs[i] for i in range(len(desired_epochs))}

    # Remove undesired keys
    models = {key : models[key] for key in desired_keys}

    print(f"Only looking at models {list(models.keys())}")
    for model_name, model_and_meta in models.items():
        model, trained = model_and_meta
        if not trained:
            print(f"will train {model_name}")
            model = model.cuda()

            epochs = desired_epochs[model_name] if model_name in desired_epochs else None
            train_loop(model, train_loader, test_loader, parameters=None, epochs=epochs)
            acc_percent = evaluate(model, test_loader)
            assert acc_percent >= 90, f"acc_percent was {acc_percent}"

            fname = os.path.join(RESNETS_FOLDER, f"{model_name}.pt")
            torch.save(model.state_dict(), fname)
    pass

# NOTE this is copied from long/resnet and modified for simplicity
def resnet18_34_stitch(snd_shape, rcv_shape):
    if type(snd_shape) == int:
        raise Exception("can't send from linear layer")
    
    snd_depth, snd_hw, _ = snd_shape
    if type(rcv_shape) == int:
        # you can pass INTO an fc
        return nn.Sequential(nn.Flatten(), nn.Linear(snd_depth * snd_hw * snd_hw, rcv_shape))
    
    # else its tensor to tensor
    rcv_depth, rcv_hw, _ = rcv_shape
    upsample_ratio = int(rcv_hw / snd_hw)
    downsample_ratio = int(snd_hw / rcv_hw)
    
    # Downsampling (or same size: 1x1) is just a strided convolution since size decreases always by a power of 2
    # every set of blocks (blocks are broken up into sets that, within those sets, have the same size).
    if downsample_ratio >= upsample_ratio:
        return nn.Conv2d(snd_depth, rcv_depth, downsample_ratio, stride=downsample_ratio, bias=True)
    else:
        return nn.Sequential(
            nn.Upsample(scale_factor=upsample_ratio, mode='nearest'),
            nn.Conv2d(snd_depth, rcv_depth, 1, stride=1, bias=True))
# NOTE that sender, reciever is format into
def resnet18_34_stitch_shape(sender, reciever):
    snd_shape, rcv_shape = None, None
    if sender == "conv1":
        snd_shape = (64, 32, 32)
    elif sender == "fc":
        raise Exception("You can't send from an FC, that's dumb")
    else:
        blockSet, _ = sender
        # every blockSet the image halves in size, the depth doubles, and the 
        ratio =  2**(blockSet - 1)
        snd_shape = (64 * ratio, 32 / ratio, 32 / ratio)
    
    if reciever == "conv1":
        rcv_shape = (3, 32, 32)
    elif reciever == "fc":
        return snd_shape, 512 # block expansion = 1 for resnet 18 and 34
    else:
        blockSet, _ = reciever
        ratio =  2**(blockSet - 1)
        rcv_shape = (64 * ratio, 32 / ratio, 32 / ratio)
    return snd_shape, rcv_shape

def resnet18_34_layer2layer():
    # NOTE format is outfrom, into
    stitches = {
        (snd_block, snd_layer) : [(rcv_block, rcv_layer) for rcv_block in range(1, 5) for rcv_layer in range(0, 2)]
        for snd_block in range(1, 5) for snd_layer in range(0, 2)
    }
    stitches["conv1"] = ["conv1"] + [(i, j) for i in range(1, 5) for j in range(0, 2)] + ["fc"]
    stitches["fc"] = ["fc"]
    # print(f"stitches are\n{stitches}")
    
    transformations = {
        (outfrom, into): resnet18_34_stitch(*resnet18_34_stitch_shape(outfrom, into))  for into in intos for outfrom, intos in stitches.items()
    }

    # TODO convert to a tabular format

    return transformations

def main_stitchtrain(args):
    print("stitch training")
    transformations = resnet18_34_layer2layer()
    print(transformations.keys())
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--name', default=None, type=str)

    parser.add_argument('--bsz', default=256, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--warmup', default=10, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--wd', default=0.01, type=float)

    parser.add_argument('--encoder', default='random-resnet18', type=str, choices=['random-resnet18',
                                                                                   'esimclr-cifar10-resnet18',
                                                                                   'esimclr-imagenet-resnet50',
                                                                                   'esimclr-cifar100-resnet18'])
    parser.add_argument('--fraction', default=1.0, type=float)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'])

    args = parser.parse_args()

    pretrain = False
    if pretrain:
        main_pretrain(args)
    else:
        raise NotImplementedError
