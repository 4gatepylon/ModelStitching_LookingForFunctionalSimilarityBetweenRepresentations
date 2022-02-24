import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from resnet import resnet18

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


def train_loop(model, train_loader, test_loader):
    parameters = []
    for layer in model:
        parameters += list(layer.parameters())

    optimizer = torch.optim.SGD(
        params=parameters,
        momentum=0.9,
        lr=args.lr * args.bsz / 256,
        weight_decay=args.wd
    )

    scaler = GradScaler()

    start = time.time()
    for e in range(1, args.epochs + 1):

        for layer in model:
            layer.train()
        # epoch
        for it, (inputs, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):

            # adjust
            adjust_learning_rate(epochs=args.epochs,
                                 warmup_epochs=args.warmup,
                                 base_lr=args.lr * args.bsz / 256,
                                 optimizer=optimizer,
                                 loader=train_loader,
                                 step=it)
            # zero grad
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                h = inputs
                for layer in model:
                    h = layer(h)
                loss = F.cross_entropy(h, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f'epoch: {e} | time: {time.time() - start:.3f}')

    eval_acc = evaluate(model, test_loader)

    print(f'final_eval_acc: {eval_acc:.3f}')

    torch.save(model[0].state_dict(), 'cifar_resnet18_supervised.pth')
    return eval_acc, model


def main(args):
    fix_seed(args.seed)

    train_loader, test_loader = get_loaders()
    model = [resnet18().cuda(), nn.Linear(512, 10).cuda()]

    train_loop(model, train_loader, test_loader)


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

    main(args)
