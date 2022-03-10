print("importing")
import os
import time
import argparse

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from resnet import RESNETS_FOLDER, resnet18, resnet34, _resnet, BasicBlock

from create_array_meta import SMALLPAIRNUM2FILENAMES

from download import (
    cifar_models_from_imagenet_models
)

from utils import fix_seed, evaluate, adjust_learning_rate, combos

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
print("done importing")

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

        print(f'\t\tepoch: {e} | time: {time.time() - start:.3f}')

    eval_acc = evaluate(model, test_loader)

    # print(f'final_eval_acc: {eval_acc:.3f}')

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
        return nn.Sequential(nn.Flatten(), nn.Linear(snd_depth * snd_hw * snd_hw, rcv_shape))#, dtype=torch.float16))
    
    # else its tensor to tensor
    rcv_depth, rcv_hw, _ = rcv_shape
    upsample_ratio = rcv_hw // snd_hw
    downsample_ratio = snd_hw // rcv_hw
    
    # Downsampling (or same size: 1x1) is just a strided convolution since size decreases always by a power of 2
    # every set of blocks (blocks are broken up into sets that, within those sets, have the same size).
    if downsample_ratio >= upsample_ratio:
        # print(f"DOWNSAMPLE {snd_shape} -> {rcv_shape}: depth={snd_depth} -> {rcv_depth}, kernel_width={downsample_ratio}, stride={downsample_ratio}")
        return nn.Conv2d(snd_depth, rcv_depth, downsample_ratio, stride=downsample_ratio, bias=True)#, dtype=torch.float16)
    else:
        return nn.Sequential(
            nn.Upsample(scale_factor=upsample_ratio, mode='nearest'),
            nn.Conv2d(snd_depth, rcv_depth, 1, stride=1, bias=True))#, dtype=torch.float16))
# NOTE that sender, reciever is format into
def resnet18_34_stitch_shape(sender, reciever):
    # print(f"SENDER IS {sender} RECIEVER IS {reciever}")
    snd_shape, rcv_shape = None, None
    if sender == "conv1":
        snd_shape = (64, 32, 32)
    elif sender == "fc":
        raise Exception("You can't send from an FC, that's dumb")
    else:
        blockSet, _ = sender
        # every blockSet the image halves in size, the depth doubles, and the 
        ratio =  2**(blockSet - 1)
        snd_shape = (64 * ratio, 32 // ratio, 32 // ratio)
    
    if reciever == "conv1":
        raise Exception("conv1 does not recieve")
    elif reciever == "fc":
        return snd_shape, 512 # block expansion = 1 for resnet 18 and 34
    else:
        # NOTE we need to give the shape that the EXPECTED SENDER gives, NOT that of the reciever
        blockSet, block = reciever
        if block == 0:
            blockSet -= 1
        if blockSet == 0:
            # It's the conv1 expectee
            return snd_shape, (64, 32, 32)
        # ^^^
        ratio =  2**(blockSet - 1)
        rcv_shape = (64 * ratio, 32 // ratio, 32 // ratio)
    return snd_shape, rcv_shape

def resnet18_34_layer2layer(sender18=True, reciever18=True):
    # look at the code in ../rumen/resnet (i.e. ./resnet)
    snd_iranges = [2, 2, 2, 2] if sender18 else [3, 4, 6, 3]
    rcv_iranges = [2, 2, 2, 2] if reciever18 else [3, 4, 6, 3]

    # 2 for conv1 and fc
    sndN = sum(snd_iranges) + 2
    rcvN = sum(rcv_iranges) + 2
    transformations = [[None for _ in range(rcvN)] for _ in range(sndN)]
    # print(f"transformations table is hxw= {sndN}x{rcvN}")

    idx2label = {}

    # Connect conv1 INTO everything else
    j = 1
    for rcv_block in range(1, 5):
        for rcv_layer in range(0, rcv_iranges[rcv_block - 1]):
            into = (rcv_block, rcv_layer)
            # print(f"[0][{j}]: conv1 -> {into}")
            transformations[0][j] = resnet18_34_stitch(*resnet18_34_stitch_shape("conv1", into))
            idx2label[(0, j)] = ("conv1", into)
            j += 1
    # print(f"[0][{j}]: conv1 -> fc")
    transformations[0][j] = resnet18_34_stitch(*resnet18_34_stitch_shape("conv1", "fc"))
    idx2label[(0, j)] = ("conv1", "fc")
    

    # Connect all the blocks INTO everything else
    i = 1
    for snd_block in range(1, 5):
        for snd_layer in range(0, snd_iranges[snd_block - 1]):
            j = 1
            outfrom = (snd_block, snd_layer)
            for rcv_block in range(1, 5):
                for rcv_layer in range(0, rcv_iranges[rcv_block - 1]):
                    into = (rcv_block, rcv_layer)
                    # print(f"[{i}][{j}]: {outfrom} -> {into}")
                    transformations[i][j] = resnet18_34_stitch(*resnet18_34_stitch_shape(outfrom, into))
                    idx2label[(i, j)] = (outfrom, into)
                    j += 1
            # print(f"[{i}][{j}]: {outfrom} -> fc")
            transformations[i][j] = resnet18_34_stitch(*resnet18_34_stitch_shape(outfrom, "fc"))
            idx2label[(i, j)] = (outfrom, "fc")
            j += 1
            i += 1
    # print(idx2label)
    return transformations, None, idx2label

class Stitched(nn.Module):
    def __init__(self, net1, net2, snd_label, rcv_label, stitch):
        super(Stitched, self).__init__()
        self.sender = net1
        self.reciever = net2
        self.snd_lbl = snd_label
        self.rcv_lbl = rcv_label
        self.stitch = stitch

    def forward(self, x):
        h = self.sender(x, vent=self.snd_lbl, into=False)
        h = self.stitch(h)
        h = self.reciever(h, vent=self.rcv_lbl, into=True)
        return h


# NOTE the use of lazy evaluation: this helps us test where we have less memory
# because it enables the garbage collector to garbage collect
def main_stitchtrain(args):
    # NOTE there are a total of 16 experiments for (r34, r18)
    # since you can pick first one r18 or r34
    # then yuo can pick the second one r18 or r34
    # then you can pick the first rand or not
    # then you can pick the second one rand or not
########################################################################################################################
    print("Generating resnet18 to resnet18 (and with random) stitches")
    resnet18_resnet18_func = lambda : resnet18_34_layer2layer(sender18=True, reciever18=True)
    resnet18_rand_resnet18_func = lambda : resnet18_34_layer2layer(sender18=True, reciever18=True)
    resnet18_resnet18_rand_func = lambda : resnet18_34_layer2layer(sender18=True, reciever18=True)
    resnet18_rand_resnet18_rand_func = lambda : resnet18_34_layer2layer(sender18=True, reciever18=True)

    print("Generating resnet18 to resnet34 (and with random) stitches")
    resnet18_resnet34_func = lambda : resnet18_34_layer2layer(sender18=True, reciever18=False)
    resnet18_rand_resnet34_func = lambda : resnet18_34_layer2layer(sender18=True, reciever18=False)
    resnet18_resnet34_rand_func = lambda : resnet18_34_layer2layer(sender18=True, reciever18=False)
    resnet18_rand_resnet34_rand_func = lambda : resnet18_34_layer2layer(sender18=True, reciever18=False)

    print("Generating resnet34 to resnet34 (and with random) stitches") 
    resnet34_resnet34_func = lambda : resnet18_34_layer2layer(sender18=False, reciever18=False)
    resnet34_resnet34_rand_func = lambda : resnet18_34_layer2layer(sender18=False, reciever18=False)
    resnet34_rand_resnet34_func = lambda : resnet18_34_layer2layer(sender18=False, reciever18=False)
    resnet34_rand_resnet34_rand_func = lambda : resnet18_34_layer2layer(sender18=False, reciever18=False)

    print("Generating resnet34 -> resnet18 with rand")
    resnet34_resnet18_func = lambda : resnet18_34_layer2layer(sender18=False, reciever18=True)
    resnet34_resnet18_rand_func = lambda : resnet18_34_layer2layer(sender18=False, reciever18=True)
    resnet34_rand_resnet18_func = lambda : resnet18_34_layer2layer(sender18=False, reciever18=True)
    resnet34_rand_resnet18_rand_func = lambda: resnet18_34_layer2layer(sender18=False, reciever18=True)
########################################################################################################################
    N = 10#len(resnet18_resnet18)
    print(f"Sims tables for resnet18 to resnet18 will be {N}x{N}")
    resnet18_resnet18_sims = [[0.0 for _ in range(N)] for _ in range(N)]
    resnet18_rand_resnet18_sims = [[0.0 for _ in range(N)] for _ in range(N)]
    resnet18_resnet18_rand_sims = [[0.0 for _ in range(N)] for _ in range(N)]
    resnet18_rand_resnet18_rand_sims = [[0.0 for _ in range(N)] for _ in range(N)]

    N = 10#len(resnet18_resnet34)
    M = 18#len(resnet18_resnet34[0])
    print(f"Sims tables for resnet18 to resnet34 will be {N}x{M}")
    resnet18_resnet34_sims = [[0.0 for _ in range(M)] for _ in range(N)]
    resnet18_rand_resnet34_sims = [[0.0 for _ in range(M)] for _ in range(N)]
    resnet18_resnet34_rand_sims = [[0.0 for _ in range(M)] for _ in range(N)]
    resnet18_rand_resnet34_rand_sims = [[0.0 for _ in range(M)] for _ in range(N)]

    N = 18#len(resnet34_resnet34)
    print(f"Sims tables for resnet34 to resnet34 will be {N}x{N}")
    resnet34_resnet34_sims = [[0.0 for _ in range(N)] for _ in range(N)]
    resnet34_resnet34_rand_sims = [[0.0 for _ in range(N)] for _ in range(N)]
    resnet34_rand_resnet34_sims = [[0.0 for _ in range(N)] for _ in range(N)]
    resnet34_rand_resnet34_rand_sims = [[0.0 for _ in range(N)] for _ in range(N)]

    N = 18#len(resnet34_resnet18)
    M = 10#len(resnet34_resnet18[0])
    print(f"Sims tables for resnet34 to resnet18 will be {N}x{M}")
    resnet34_resnet18_sims = [[0.0 for _ in range(M)] for _ in range(N)]
    resnet34_resnet18_rand_sims = [[0.0 for _ in range(M)] for _ in range(N)]
    resnet34_rand_resnet18_sims = [[0.0 for _ in range(M)] for _ in range(N)]
    resnet34_rand_resnet18_rand_sims = [[0.0 for _ in range(M)] for _ in range(N)]
########################################################################################################################
    print("Loading resnets into memory from disk")
    r18 = resnet18().cuda()
    r18.load_state_dict(torch.load(os.path.join(RESNETS_FOLDER, "resnet18.pt")))#, map_location=torch.device('cpu')))
    r18_rand = resnet18().cuda()
    r34 = resnet34().cuda()
    r34.load_state_dict(torch.load(os.path.join(RESNETS_FOLDER, "resnet34.pt")))#, map_location=torch.device('cpu')))
    r34_rand = resnet34().cuda()

    print("Getting loaders")
    train_loader, test_loader = get_loaders()

    print("Confirming that acc is high")
    accp = 0.0
    accp = evaluate(r18, test_loader)
    r18_acc = accp / 100.0
    print(f"Accuracy of resnet18 is {accp}%")
    assert accp > 90
    
    accp = evaluate(r18_rand, test_loader)
    r18_rand_acc = accp / 100.0
    print(f"Accuracy of resnet18 random is {accp}%")
    assert accp < 20

    accp = evaluate(r34, test_loader)
    r34_acc = accp / 100.0
    print(f"Accuracy of resnet34 is {accp}%")
    assert accp > 90

    accp = evaluate(r34_rand, test_loader)
    r34_rand_acc = accp / 100.0
    print(f"Accuracy of resnet34 random is {accp}%")
    assert accp < 20
########################################################################################################################
    print("Generating similarity tables")

    folder = f"sims{args.fnum}"
    if not os.path.exists(folder):
            os.mkdir(folder)

    for sender, reciever, transformations_func, table, orig_acc1, orig_acc2, name in [
        # Resnet18 -> Resnet18
        (r18, r18, resnet18_resnet18_func, resnet18_resnet18_sims, r18_acc, r18_acc, "resnet18_resnet18"),
        (r18_rand, r18, resnet18_rand_resnet18_func, resnet18_rand_resnet18_sims,  r18_acc, r18_rand_acc, "resnet18_rand_resnet18"),
        (r18, r18_rand, resnet18_resnet18_rand_func, resnet18_resnet18_rand_sims, r18_acc, r18_rand_acc, "resnet18_resnet18_rand"),
        (r18_rand, r18_rand, resnet18_rand_resnet18_rand_func, resnet18_rand_resnet18_rand_sims, r18_rand_acc, r18_rand_acc, "resnet18_rand_resnet18_rand"),

        # Resnet18 -> Resnet34
        (r18, r34, resnet18_resnet34_func, resnet18_resnet34_sims, r18_acc, r34_acc, "resnet18_resnet34"),
        (r18_rand, r34, resnet18_rand_resnet34_func, resnet18_rand_resnet34_sims, r18_rand_acc, r34_acc,"resnet18_rand_resnet34"),
        (r18_rand, r34_rand, resnet18_rand_resnet34_rand_func, resnet18_rand_resnet34_rand_sims, r18_rand_acc, r34_rand_acc, "resnet18_rand_resnet34_rand"),
        (r18, r34_rand, resnet18_resnet34_rand_func, resnet18_resnet34_rand_sims, r18_acc, r34_rand_acc,"resnet18_resnet34_rand"),
        
        # Resnet34 -> Resnet18
        (r34, r18, resnet34_resnet18_func, resnet34_resnet18_sims, r34_acc, r18_acc, "resnet34_resnet18"),
        (r34_rand, r18, resnet34_rand_resnet18_func, resnet34_rand_resnet18_sims, r34_rand_acc, r18_acc, "resnet34_rand_resnet18"),
        (r34, r18_rand, resnet34_resnet18_rand_func, resnet34_resnet18_rand_sims, r34_acc, r18_rand_acc, "resnet34_resnet18_rand"),
        (r34_rand, r18_rand, resnet34_rand_resnet18_rand_func, resnet34_rand_resnet18_rand_sims, r34_rand_acc, r18_rand_acc, "resnet34_rand_resnet18_rand"),
        
        # Resnet34 -> Resnet34
        (r34, r34, resnet34_resnet34_func, resnet34_resnet34_sims, r34_acc, r34_acc, "resnet34_resnet34"),
        (r34_rand, r34, resnet34_rand_resnet34_func, resnet34_rand_resnet34_sims, r34_rand_acc, r34_acc, "resnet34_rand_resnet34"),
        (r34, r34_rand, resnet34_resnet34_rand_func, resnet34_resnet34_rand_sims, r34_acc, r34_rand_acc, "resnet34_resnet34_rand"),
        (r34_rand, r34_rand, resnet34_rand_resnet34_rand_func, resnet34_rand_resnet34_rand_sims, r34_rand_acc, r34_rand_acc, "resnet34_rand_resnet34_rand"),
        ]:
        
        N = len(table)
        M = len(table[0])
        transformations, _, idx2label = transformations_func()

        # NOTE we ignore first and last layer (for reciever and send)
        # NOTE outfrom, into format
        for i in range(N - 1):
            for j in range(1, M):
                try:
                    acc = 0.0
                    orig_acc = min(orig_acc1, orig_acc2)

                    snd_label, rcv_label = idx2label[(i, j)]
                    print(f"\tstitching {i}->{j} which is {snd_label}->{rcv_label}")
                    st = transformations[i][j].cuda()
                    model = Stitched(sender, reciever, snd_label, rcv_label, st)
                    acc, _ = train_loop(model, train_loader, test_loader, epochs=4, parameters=list(st.parameters()))
                    acc /= 100.0
                    print(f"\tAccuracy of model is {acc}")
                    print(f"\tOriginal accuracy was {orig_acc}")
                    table[i][j] = acc
                    
                    filename_stitch = name + f"{i}_{j}_stitch.pt"
                    filename_stitch = os.path.join(folder, filename_stitch)
                    print(f"\tSaving stitch {i}->{j} to filename {filename_stitch}")
                    torch.save(transformations[i][j].cpu().state_dict(), filename_stitch)
                except:
                    table[i][j] = -1
        
        
        filename_sims = name + "_sims.pt"
        filename_sims = os.path.join(folder, filename_sims)
        print(f"Saving sims to {filename_sims}")
        torch.save(torch.tensor(table), filename_sims)
########################################################################################################################
    pass

# Train a bunch of small resnets: all combinations from 1, 1, 1, 1 to 2, 2, 2, 2
# (there are 16 combinations)
def main_train_small(args):
    train_loader, test_loader = get_loaders()

    combinations = combos(4, [1, 2])
    for combination in combinations:
        x, y, z, w = combination
        fname = os.path.join(RESNETS_FOLDER, f"resnet_{x}{y}{z}{w}.pt")
        if os.path.exists(fname):
            print(f"Skipping {fname}")
            continue
        else:
            # NOT pretrained and NO progress bar (these aren't supported anyways)
            model = _resnet(f"resnet_{x}{y}{z}{w}", BasicBlock, combination, False, False)
            model = model.cuda()

            print(f"will train net {x}{y}{z}{w}")

            # TODO play around with epochs
            epochs = 40
            train_loop(model, train_loader, test_loader, parameters=None, epochs=epochs)
            acc_percent = evaluate(model, test_loader)
            print(f"acc_percent was {acc_percent}")

            # NOTE this is ../pretrained_resnet
            torch.save(model.state_dict(), fname)
    pass

def resnet_small_small_layer2layer(snd_iranges, rcv_iranges):
    N1 = sum(snd_iranges) + 2
    N2 = sum(rcv_iranges) + 2
    transformations = [[None for _ in range(N2)] for _ in range(N1)]
    idx2label = {}

    # NOTE this is copied and modified from resnet_18_34_layer2layer
    # (yea I'm pretty lazy LOL)

    # NOTE in theory the "resnet18_34_stitch" and "resnet_18_34_stitch_shape"
    # functions SHOULD work here regardless of the fact that they are in fact not meant for it
    # (they should work for any basic block [x, y, z, w])

    # Connect conv1 INTO everything else
    j = 1
    for rcv_block in range(1, 5):
        for rcv_layer in range(0, rcv_iranges[rcv_block - 1]):
            into = (rcv_block, rcv_layer)
            # print(f"[0][{j}]: conv1 -> {into}")
            transformations[0][j] = resnet18_34_stitch(*resnet18_34_stitch_shape("conv1", into))
            idx2label[(0, j)] = ("conv1", into)
            j += 1
    # print(f"[0][{j}]: conv1 -> fc")
    transformations[0][j] = resnet18_34_stitch(*resnet18_34_stitch_shape("conv1", "fc"))
    idx2label[(0, j)] = ("conv1", "fc")
    

    # Connect all the blocks INTO everything else
    i = 1
    for snd_block in range(1, 5):
        for snd_layer in range(0, snd_iranges[snd_block - 1]):
            j = 1
            outfrom = (snd_block, snd_layer)
            for rcv_block in range(1, 5):
                for rcv_layer in range(0, rcv_iranges[rcv_block - 1]):
                    into = (rcv_block, rcv_layer)
                    # print(f"[{i}][{j}]: {outfrom} -> {into}")
                    transformations[i][j] = resnet18_34_stitch(*resnet18_34_stitch_shape(outfrom, into))
                    idx2label[(i, j)] = (outfrom, into)
                    j += 1
            # print(f"[{i}][{j}]: {outfrom} -> fc")
            transformations[i][j] = resnet18_34_stitch(*resnet18_34_stitch_shape(outfrom, "fc"))
            idx2label[(i, j)] = (outfrom, "fc")
            j += 1
            i += 1
    return transformations, None, idx2label

# NOTE: we use this shit to be parallel
# https://supercloud.mit.edu/submitting-jobs#slurm-gpus
def main_stitchtrain_small(args):
    file1, file2 = SMALLPAIRNUM2FILENAMES[int(args.smallpairnum)]
    numbers1 = list(map(int, file1.split(".")[0][-4:]))
    numbers2 = list(map(int, file2.split(".")[0][-4:]))
    name1 = "resnet_" + "".join(map(str, numbers1))
    name2 = "resnet_" + "".join(map(str, numbers2))
    output_folder = f"sims_{name1}_{name2}"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print(f"Will stitch {file1} and {file2} in {RESNETS_FOLDER}")
    print(f"numbers1: {numbers1}")
    print(f"numbers2: {numbers2}")
    print(f"name1: {name1}")
    print(f"name2: {name2}")

    file1 = os.path.join(RESNETS_FOLDER, file1)
    file2 = os.path.join(RESNETS_FOLDER, file2)

    print("Loading models")
    model1 = _resnet(name1, BasicBlock, numbers1, False, False)
    model2 = _resnet(name2, BasicBlock, numbers2, False, False)
    model1_rand = _resnet(name1 + "_rand", BasicBlock, numbers1, False, False)
    model2_rand = _resnet(name2 + "_rand", BasicBlock, numbers2, False, False)
    model1.cuda()
    model2.cuda()
    model1_rand.cuda()
    model2_rand.cuda()
    model1.load_state_dict(torch.load(file1))
    model2.load_state_dict(torch.load(file2))

    print("Getting loaders")
    train_loader, test_loader = get_loaders()

    print("Evaluating accuracies of pretrained models")
    accp = 0.0
    accp = evaluate(model1, test_loader)
    model1_acc = accp / 100.0
    print(f"Accuracy of {name1} is {accp}%")
    assert accp > 90
    
    accp = evaluate(model1_rand, test_loader)
    model1_rand_acc = accp / 100.0
    print(f"Accuracy of {name1} random is {accp}%")
    assert accp < 20

    accp = evaluate(model2, test_loader)
    model2_acc = accp / 100.0
    print(f"Accuracy of {name2} is {accp}%")
    assert accp > 90

    accp = evaluate(model2_rand, test_loader)
    model2_rand_acc = accp / 100.0
    print(f"Accuracy of {name2} random is {accp}%")
    assert accp < 20

    # Number of blocks/layers to stitch in model 1 and model 2
    N1 = sum(numbers1) + 2
    N2 = sum(numbers2) + 2

    model1_model2_sims = [[0.0 for _ in range(N2)] for _ in range(N1)]
    model1_rand_model2_sims = [[0.0 for _ in range(N2)] for _ in range(N1)]
    model1_model2_rand_sims = [[0.0 for _ in range(N2)] for _ in range(N1)]
    model1_rand_model2_rand_sims = [[0.0 for _ in range(N2)] for _ in range(N1)]

    print("Initializing the stitches (note all idx2label are the same)")
    model1_model2_stitches, _, idx2label = resnet_small_small_layer2layer(numbers1, numbers2)
    model1_rand_model2_stitches, _, _ = resnet_small_small_layer2layer(numbers1, numbers2)
    model1_model2_rand_stitches, _, _ = resnet_small_small_layer2layer(numbers1, numbers2)
    model1_rand_model2_rand_stitches, _, _ = resnet_small_small_layer2layer(numbers1, numbers2)

    pp.pprint(idx2label)
    
    print(f"Stitching, will save in {output_folder}")
    for sender, reciever, stitches, sims, orig_acc1, orig_acc2, name in [
        # Resnet18 -> Resnet18
        (model1, model2, model1_model2_stitches, model1_model2_sims, model1_acc, model2_acc, f"{name1}_{name2}"),
        (model1_rand, model2, model1_rand_model2_stitches, model1_rand_model2_sims, model1_rand_acc, model2_acc, f"{name1}_rand_{name2}"),
        (model1, model2_rand, model1_model2_rand_stitches, model1_model2_rand_sims, model1_acc, model2_rand_acc, f"{name1}_{name2}_rand"),
        (model1_rand, model2_rand, model1_rand_model2_rand_stitches, model1_rand_model2_rand_sims, model1_rand_acc, model2_rand_acc, f"{name1}_rand_{name2}_rand"),
    ]:
        print(f"Stitching {name}")
        # NOTE we ignore first and last layer (for reciever and send)
        # NOTE outfrom, into format
        for i in range(N1 - 1):
            for j in range(1, N2):
                try:
                    acc = 0.0

                    snd_label, rcv_label = idx2label[(i, j)]
                    print(f"\tstitching {i}->{j} which is {snd_label}->{rcv_label}")

                    stitch = stitches[i][j].cuda()
                    model = Stitched(sender, reciever, snd_label, rcv_label, stitch)

                    # Epochs can be changed (large may lead to exploding gradients?)
                    acc, _ = train_loop(model, train_loader, test_loader, epochs=3, parameters=list(stitch.parameters()))
                    acc /= 100.0
                    print(f"\tAccuracy of model is {acc}")
                    print(f"\tOriginal accuracies were {orig_acc1} and {orig_acc2}")
                    sims[i][j] = acc
                except Exception as e:
                    print(f"Failed on layers {i}, {j} ({e})")
                    sims[i][j] = -1
        
        filename_sims = name + "_sims.pt"
        filename_sims = os.path.join(output_folder, filename_sims)
        print(f"Saving sims to {filename_sims}")
        torch.save(torch.tensor(sims), filename_sims)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--name', default=None, type=str)

    parser.add_argument('--bsz', default=256, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--warmup', default=10, type=int)
    parser.add_argument('--epochs', default=1, type=int) # NOTE this is modified
    parser.add_argument('--wd', default=0.01, type=float)

    parser.add_argument('--encoder', default='random-resnet18', type=str, choices=['random-resnet18',
                                                                                   'esimclr-cifar10-resnet18',
                                                                                   'esimclr-imagenet-resnet50',
                                                                                   'esimclr-cifar100-resnet18'])
    parser.add_argument('--fraction', default=1.0, type=float)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--fnum', default='0', type=str) # Used by Resnet18 - Resnet34 experiments to do multiple copies with an id
    parser.add_argument('--smallpairnum', default='1', type=str) # Used by Resnet 1111 -> 2222 to select which pair num to use
    args = parser.parse_args()

    main_stitchtrain_small(args)
    # main_train_small(args)
    # pretrain = False
    # if pretrain:
    #     main_pretrain(args)
    # else:
    #     main_stitchtrain(args)

# Resnet34 dimensions for batch size 1:
# NOTE how they are THE SAME AS THOSE OF RESNET18! (within block sets)
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# ResNet                                   --                        --
# ├─Conv2d: 1-1                            [1, 64, 32, 32]           1,728
# ├─BatchNorm2d: 1-2                       [1, 64, 32, 32]           128
# ├─ReLU: 1-3                              [1, 64, 32, 32]           --
# ├─Sequential: 1-4                        [1, 64, 32, 32]           --
# │    └─BasicBlock: 2-1                   [1, 64, 32, 32]           --
# │    │    └─Conv2d: 3-1                  [1, 64, 32, 32]           36,864
# │    │    └─BatchNorm2d: 3-2             [1, 64, 32, 32]           128
# │    │    └─ReLU: 3-3                    [1, 64, 32, 32]           --
# │    │    └─Conv2d: 3-4                  [1, 64, 32, 32]           36,864
# │    │    └─BatchNorm2d: 3-5             [1, 64, 32, 32]           128
# │    │    └─ReLU: 3-6                    [1, 64, 32, 32]           --
# │    └─BasicBlock: 2-2                   [1, 64, 32, 32]           --
# │    │    └─Conv2d: 3-7                  [1, 64, 32, 32]           36,864
# │    │    └─BatchNorm2d: 3-8             [1, 64, 32, 32]           128
# │    │    └─ReLU: 3-9                    [1, 64, 32, 32]           --
# │    │    └─Conv2d: 3-10                 [1, 64, 32, 32]           36,864
# │    │    └─BatchNorm2d: 3-11            [1, 64, 32, 32]           128
# │    │    └─ReLU: 3-12                   [1, 64, 32, 32]           --
# │    └─BasicBlock: 2-3                   [1, 64, 32, 32]           --
# │    │    └─Conv2d: 3-13                 [1, 64, 32, 32]           36,864
# │    │    └─BatchNorm2d: 3-14            [1, 64, 32, 32]           128
# │    │    └─ReLU: 3-15                   [1, 64, 32, 32]           --
# │    │    └─Conv2d: 3-16                 [1, 64, 32, 32]           36,864
# │    │    └─BatchNorm2d: 3-17            [1, 64, 32, 32]           128
# │    │    └─ReLU: 3-18                   [1, 64, 32, 32]           --
# ├─Sequential: 1-5                        [1, 128, 16, 16]          --
# │    └─BasicBlock: 2-4                   [1, 128, 16, 16]          --
# │    │    └─Conv2d: 3-19                 [1, 128, 16, 16]          73,728
# │    │    └─BatchNorm2d: 3-20            [1, 128, 16, 16]          256
# │    │    └─ReLU: 3-21                   [1, 128, 16, 16]          --
# │    │    └─Conv2d: 3-22                 [1, 128, 16, 16]          147,456
# │    │    └─BatchNorm2d: 3-23            [1, 128, 16, 16]          256
# │    │    └─Sequential: 3-24             [1, 128, 16, 16]          8,448
# │    │    └─ReLU: 3-25                   [1, 128, 16, 16]          --
# │    └─BasicBlock: 2-5                   [1, 128, 16, 16]          --
# │    │    └─Conv2d: 3-26                 [1, 128, 16, 16]          147,456
# │    │    └─BatchNorm2d: 3-27            [1, 128, 16, 16]          256
# │    │    └─ReLU: 3-28                   [1, 128, 16, 16]          --
# │    │    └─Conv2d: 3-29                 [1, 128, 16, 16]          147,456
# │    │    └─BatchNorm2d: 3-30            [1, 128, 16, 16]          256
# │    │    └─ReLU: 3-31                   [1, 128, 16, 16]          --
# │    └─BasicBlock: 2-6                   [1, 128, 16, 16]          --
# │    │    └─Conv2d: 3-32                 [1, 128, 16, 16]          147,456
# │    │    └─BatchNorm2d: 3-33            [1, 128, 16, 16]          256
# │    │    └─ReLU: 3-34                   [1, 128, 16, 16]          --
# │    │    └─Conv2d: 3-35                 [1, 128, 16, 16]          147,456
# │    │    └─BatchNorm2d: 3-36            [1, 128, 16, 16]          256
# │    │    └─ReLU: 3-37                   [1, 128, 16, 16]          --
# │    └─BasicBlock: 2-7                   [1, 128, 16, 16]          --
# │    │    └─Conv2d: 3-38                 [1, 128, 16, 16]          147,456
# │    │    └─BatchNorm2d: 3-39            [1, 128, 16, 16]          256
# │    │    └─ReLU: 3-40                   [1, 128, 16, 16]          --
# │    │    └─Conv2d: 3-41                 [1, 128, 16, 16]          147,456
# │    │    └─BatchNorm2d: 3-42            [1, 128, 16, 16]          256
# │    │    └─ReLU: 3-43                   [1, 128, 16, 16]          --
# ├─Sequential: 1-6                        [1, 256, 8, 8]            --
# │    └─BasicBlock: 2-8                   [1, 256, 8, 8]            --
# │    │    └─Conv2d: 3-44                 [1, 256, 8, 8]            294,912
# │    │    └─BatchNorm2d: 3-45            [1, 256, 8, 8]            512
# │    │    └─ReLU: 3-46                   [1, 256, 8, 8]            --
# │    │    └─Conv2d: 3-47                 [1, 256, 8, 8]            589,824
# │    │    └─BatchNorm2d: 3-48            [1, 256, 8, 8]            512
# │    │    └─Sequential: 3-49             [1, 256, 8, 8]            33,280
# │    │    └─ReLU: 3-50                   [1, 256, 8, 8]            --
# │    └─BasicBlock: 2-9                   [1, 256, 8, 8]            --
# │    │    └─Conv2d: 3-51                 [1, 256, 8, 8]            589,824
# │    │    └─BatchNorm2d: 3-52            [1, 256, 8, 8]            512
# │    │    └─ReLU: 3-53                   [1, 256, 8, 8]            --
# │    │    └─Conv2d: 3-54                 [1, 256, 8, 8]            589,824
# │    │    └─BatchNorm2d: 3-55            [1, 256, 8, 8]            512
# │    │    └─ReLU: 3-56                   [1, 256, 8, 8]            --
# │    └─BasicBlock: 2-10                  [1, 256, 8, 8]            --
# │    │    └─Conv2d: 3-57                 [1, 256, 8, 8]            589,824
# │    │    └─BatchNorm2d: 3-58            [1, 256, 8, 8]            512
# │    │    └─ReLU: 3-59                   [1, 256, 8, 8]            --
# │    │    └─Conv2d: 3-60                 [1, 256, 8, 8]            589,824
# │    │    └─BatchNorm2d: 3-61            [1, 256, 8, 8]            512
# │    │    └─ReLU: 3-62                   [1, 256, 8, 8]            --
# │    └─BasicBlock: 2-11                  [1, 256, 8, 8]            --
# │    │    └─Conv2d: 3-63                 [1, 256, 8, 8]            589,824
# │    │    └─BatchNorm2d: 3-64            [1, 256, 8, 8]            512
# │    │    └─ReLU: 3-65                   [1, 256, 8, 8]            --
# │    │    └─Conv2d: 3-66                 [1, 256, 8, 8]            589,824
# │    │    └─BatchNorm2d: 3-67            [1, 256, 8, 8]            512
# │    │    └─ReLU: 3-68                   [1, 256, 8, 8]            --
# │    └─BasicBlock: 2-12                  [1, 256, 8, 8]            --
# │    │    └─Conv2d: 3-69                 [1, 256, 8, 8]            589,824
# │    │    └─BatchNorm2d: 3-70            [1, 256, 8, 8]            512
# │    │    └─ReLU: 3-71                   [1, 256, 8, 8]            --
# │    │    └─Conv2d: 3-72                 [1, 256, 8, 8]            589,824
# │    │    └─BatchNorm2d: 3-73            [1, 256, 8, 8]            512
# │    │    └─ReLU: 3-74                   [1, 256, 8, 8]            --
# │    └─BasicBlock: 2-13                  [1, 256, 8, 8]            --
# │    │    └─Conv2d: 3-75                 [1, 256, 8, 8]            589,824
# │    │    └─BatchNorm2d: 3-76            [1, 256, 8, 8]            512
# │    │    └─ReLU: 3-77                   [1, 256, 8, 8]            --
# │    │    └─Conv2d: 3-78                 [1, 256, 8, 8]            589,824
# │    │    └─BatchNorm2d: 3-79            [1, 256, 8, 8]            512
# │    │    └─ReLU: 3-80                   [1, 256, 8, 8]            --
# ├─Sequential: 1-7                        [1, 512, 4, 4]            --
# │    └─BasicBlock: 2-14                  [1, 512, 4, 4]            --
# │    │    └─Conv2d: 3-81                 [1, 512, 4, 4]            1,179,648
# │    │    └─BatchNorm2d: 3-82            [1, 512, 4, 4]            1,024
# │    │    └─ReLU: 3-83                   [1, 512, 4, 4]            --
# │    │    └─Conv2d: 3-84                 [1, 512, 4, 4]            2,359,296
# │    │    └─BatchNorm2d: 3-85            [1, 512, 4, 4]            1,024
# │    │    └─Sequential: 3-86             [1, 512, 4, 4]            132,096
# │    │    └─ReLU: 3-87                   [1, 512, 4, 4]            --
# │    └─BasicBlock: 2-15                  [1, 512, 4, 4]            --
# │    │    └─Conv2d: 3-88                 [1, 512, 4, 4]            2,359,296
# │    │    └─BatchNorm2d: 3-89            [1, 512, 4, 4]            1,024
# │    │    └─ReLU: 3-90                   [1, 512, 4, 4]            --
# │    │    └─Conv2d: 3-91                 [1, 512, 4, 4]            2,359,296
# │    │    └─BatchNorm2d: 3-92            [1, 512, 4, 4]            1,024
# │    │    └─ReLU: 3-93                   [1, 512, 4, 4]            --
# │    └─BasicBlock: 2-16                  [1, 512, 4, 4]            --
# │    │    └─Conv2d: 3-94                 [1, 512, 4, 4]            2,359,296
# │    │    └─BatchNorm2d: 3-95            [1, 512, 4, 4]            1,024
# │    │    └─ReLU: 3-96                   [1, 512, 4, 4]            --
# │    │    └─Conv2d: 3-97                 [1, 512, 4, 4]            2,359,296
# │    │    └─BatchNorm2d: 3-98            [1, 512, 4, 4]            1,024
# │    │    └─ReLU: 3-99                   [1, 512, 4, 4]            --
# ├─AdaptiveAvgPool2d: 1-8                 [1, 512, 1, 1]            --
# ==========================================================================================
# Total params: 21,276,992
# Trainable params: 21,276,992
# Non-trainable params: 0
# Total mult-adds (G): 1.16
# ==========================================================================================
# Input size (MB): 0.01
# Forward/backward pass size (MB): 16.38
# Params size (MB): 85.11
# Estimated Total Size (MB): 101.50
# ==========================================================================================