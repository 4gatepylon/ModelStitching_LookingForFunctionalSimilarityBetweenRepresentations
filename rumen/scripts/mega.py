# A Massive script to try and do the entire stitching experiment (in its simplest possible form)
# without any external context.

from ffcv.writer import (
    DatasetWriter,
)
from ffcv.transforms.common import (
    Squeeze,
)
from ffcv.transforms import (
    RandomHorizontalFlip,
    Cutout,
    RandomTranslate,
    Convert,
    ToDevice,
    ToTensor,
    ToTorchImage,
)
from ffcv.pipeline.operation import (
    Operation,
)
from ffcv.loader import (
    Loader,
    OrderOption,
)
from ffcv.fields.decoders import (
    IntDecoder,
    SimpleRGBImageDecoder,
    RandomResizedCropRGBImageDecoder,
)
from ffcv.fields import (
    IntField,
    RGBImageField,
)

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.cuda.amp import GradScaler, autocast

from mega_resnet import BasicBlock
from mega_resnet import make_resnet
from mega_resnet import make_stitched_resnet

# NOTE same as the original because it's the same tree height away
RESNETS_FOLDER = "../../pretrained_resnets/"
SIMS_FOLDER = "../../sims/"
HEATMAPS_FOLDER = "../../heatmaps/"

# Loaders folders
FFCV_FOLDER = "../../data_ffcv/"
MISC_FFCV_FOLDER = "../tmp/"

FFCV_CIFAR_MEAN = [125.307, 122.961, 113.8575]
FFCV_CIFAR_STD = [51.5865, 50.847, 51.255]
FFCV_NORMALIZE_TRANSFORM = torchvision.transforms.Normalize(
    FFCV_CIFAR_MEAN, FFCV_CIFAR_STD)

def pclone(model):
    return [p.data.detach().clone() for p in model.parameters()]


def listeq(l1, l2):
    return min((torch.eq(a, b).int().min().item() for a, b in zip(l1, l2))) == 1

def num_labels(numbers):
    # input, conv1, ... one for each block, blockset, ... fc, output
    return 2 + 2 + sum(numbers)

# NOTE: tick labels for y before x because y is the sender
def matrix_heatmap(input_file_name: str, output_file_name: str, tick_lables_y=None, tick_labels_x=None):
        mat = torch.load(input_file_name)
        assert type(mat) == torch.Tensor or type(mat) == np.ndarray
        if type(mat) == torch.Tensor:
            mat = mat.numpy()
        assert len(mat.shape) == 2

        mat_height, mat_width = mat.shape

        # NOTE: rounding is ugly for mean squared errors (but it's great for sim matrices)
        mat = np.round(mat, decimals=2)

        yticks, xticks = np.arange(mat_height), np.arange(mat_width)
        if tick_labels_y:
            assert len(tick_labels_y) == mat_height
        else:
            tick_labels_y = yticks
        if tick_labels_x
            assert len(tick_labels_x) == mat_width
        else:
            tick_labels_x = xticks
       

        fig, ax = plt.subplots()
        im = ax.imshow(mat)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels(tick_labels_x)
        ax.set_yticklabels(tick_labels_y)
        plt.setp(ax.get_xticklabels(), rotation=45,
                 ha="right", rotation_mode="anchor")

        # This is inserting the text into the boxes so that we can compare easily
        for i in range(len(yticks)):
            for j in range(len(xticks)):
                text = ax.text(j, i, mat[i, j],
                               ha="center", va="center", color="w")

        title = input_file_name.split("/")[-1].split('.')[0]
        ax.set_title(f"{title}")
        fig.tight_layout()
        plt.savefig(output_file_name)
        plt.clf()

def get_loaders(args):
    num_of_points = 50000
    split = [int(num_of_points * args.fraction),
                int(num_of_points * (1 - args.fraction))]

    dataset_class = torchvision.datasets.CIFAR10 if (
        args.dataset == 'cifar10') else torchvision.datasets.CIFAR100
    finetune_file = os.path.join(
        MISC_FFCV_FOLDER, f"finetune_{args.dataset}_{args.fraction}_train_data.beton")
    if not os.path.exists(finetune_file):
        train_data = dataset_class(
            FFCV_FOLDER, train=True, download=True
        )
        train_data = torch.utils.data.random_split(train_data, split)[0]
        train_writer = DatasetWriter(finetune_file, {
            'image': RGBImageField(),
            'label': IntField()
        })
        train_writer.from_indexed_dataset(train_data)

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        ToDevice('cuda:0'),
        Squeeze(),
    ]

    image_pipeline_train: List[Operation] = [
        SimpleRGBImageDecoder(),
        RandomHorizontalFlip(),
        RandomTranslate(padding=2, fill=tuple(map(int, FFCV_CIFAR_MEAN))),
        Cutout(4, tuple(map(int, FFCV_CIFAR_MEAN))),
        ToTensor(),
        ToDevice('cuda:0', non_blocking=True),
        ToTorchImage(),
        Convert(torch.float16),
        FFCV_NORMALIZE_TRANSFORM
    ]
    train_loader = Loader(finetune_file,
                            batch_size=args.bsz,
                            num_workers=args.num_workers,
                            order=OrderOption.RANDOM,
                            os_cache=True,
                            drop_last=True,
                            pipelines={
                                'image': image_pipeline_train,
                                'label': label_pipeline,
                            })

    test_data_file = os.path.join(
        MISC_FFCV_FOLDER, f"{args.dataset}_test_data.beton")
    if not os.path.exists(test_data_file):
        test_data = dataset_class(
            FFCV_FOLDER, train=False, download=True
        )

        test_writer = DatasetWriter(test_data_file, {
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
        torchvision.transforms.Normalize(FFCV_CIFAR_MEAN, FFCV_CIFAR_STD)
    ]

    test_loader = Loader(test_data_file,
                            batch_size=2048,
                            num_workers=args.num_workers,
                            order=OrderOption.SEQUENTIAL,
                            os_cache=True,
                            drop_last=False,
                            pipelines={
                                'image': image_pipeline_test,
                                'label': label_pipeline,
                            })
    return train_loader, test_loader

def adjust_learning_rate(
    epochs,
    warmup_epochs,
    base_lr,
    optimizer,
    loader,
    step,
):
    max_steps = epochs * len(loader)
    warmup_steps = warmup_epochs * len(loader)
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = 0
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def evaluate(model, test_loader):
    # NOTE used to be for layer in model
    model.eval()

    for _, (images, labels) in enumerate(test_loader):
        total_correct, total_num = 0., 0.

        with torch.no_grad():
            with autocast():
                h = images
                h = model(h)
                preds = h.argmax(dim=1)
                total_correct = (preds == labels).sum().cpu().item()
                total_num += h.shape[0]

    return total_correct / total_num

def train_loop(
    args,
    model,
    train_loader,
    test_loader,
    verbose = True,
):
    parameters = list(model.parameters())

    optimizer = torch.optim.SGD(
        params=parameters,
        momentum=0.9,
        lr=args.lr * args.bsz / 256,
        weight_decay=args.wd
    )

    scaler = GradScaler()

    start = time.time()
    epochs = args.epochs
    for e in range(1, epochs + 1):
        if verbose:
            print(f"\t\t starting on epoch {e} for {len(train_loader)} iterations")
        model.train()
        # epoch
        # NOTE that enumerate's start changes the starting index
        for it, (inputs, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
            # TODO not sure why it's so slow sometimes, but it seems to need to "Warm up"
            # ... I've never seen this before ngl
            # print(f"\t\t\titeration {it}")
            # adjust
            adjust_learning_rate(epochs=epochs,
                                    warmup_epochs=args.warmup,
                                    base_lr=args.lr * args.bsz / 256,
                                    optimizer=optimizer,
                                    loader=train_loader,
                                    step=it)
            # zero grad (should we set to none?)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                h = inputs
                h = model(h)
                #print(h)
                #print(y)
                # TODO modularize this out to enable sim training
                loss = F.cross_entropy(h, y)
                #print(loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if verbose:
            print(f'\t\tepoch: {e} | time: {time.time() - start:.3f}')

    eval_acc = evaluate(model, test_loader)

    return eval_acc

def get_shape(label):
    if label == "input":
        return (3, 32, 32)
    elif label == "conv1":
        return (64, 32, 32)
    elif label == "fc":
        return 10
    elif label == "output":
        raise Exception("output layer has no output shape")
    if type(label) == tuple:
        blockset, _ = label
        if blockset == 1:
            return (64, 32, 32)
        elif blockset == 2:
            return (128, 16, 16)
        elif blockset == 3:
            return (256, 8, 8)
        elif blockset == 4:
            return (512, 4, 4)
    else:
        raise Exception(f"unknown layer type in get_shape: {label}")

# Tells you the shape that's coming INTO a layer
# Which is either a 3-tuple for tensors or a single scalar for the width
# of a linear layer
def get_prev_shape(label):
    if label == "input":
        raise Exception("input layer has no previous shape")
    if label == "conv1":
        return (3, 32, 32)
    if label == "fc":
        return (512, 4, 4)
    if label == "output":
        return 10
    if type(label) == tuple:
        blockset, _ = label
        if blockset == 1 or blockset == 2:
            return (64, 32, 32)
        if blockset == 3:
            return (128, 16, 16)
        if blockset == 4:
            return (256, 8, 8)
        else:
            raise Exception(f"unknown tuple label {label} in get_prev_shape")
    else:
        raise Exception(f"unknown label: {label} in get_prev_shape")

def make_stitch(send_label, recv_label):
    send_shape = get_shape(send_label)
    recv_shape = get_prev_shape(recv_label)
    assert type(send_shape) == tuple
    assert len(send_shape) == 3
    # assert type(recv_shape) == tuple or t
    if type(recv_shape) == int:
        return nn.Sequential(
        nn.Flatten(),
        nn.Linear(
            send_shape[0] * send_shape[1] * send_shape[2],
            recv_shape,
        ),
    )
    else:
        send_depth, send_height, send_width = send_shape
        recv_depth, recv_height, recv_width = recv_shape
        if recv_height <= send_height:
            ratio = send_height // recv_height
            return nn.Conv2d(
                send_depth,
                recv_depth,
                ratio,
                stride=ratio,
                bias=True,
            )
        else:
            ratio = recv_height // send_height
            return nn.Sequential(
                nn.Upsample(
                    scale_factor=ratio,
                    mode='nearest',
                ),
                nn.Conv2d(
                    send_depth,
                    recv_depth,
                    1,
                    stride=1,
                    bias=True,
                ),
            )

# NOTE that this code
# 1. does NOT pretrain (i.e. regular train) the models...
#    instead you can use `experiment.py` for that
# 2. does NOT use controls (i.e. random networks) since
#    we've more or less used them before and want to get
#    a quick sanity test batch of results
def stitchtrain(numbers1, numbers2, args):
    name1 = f"resnet_{"".join(map(str, numbers1))}"
    name2 = f"resnet_{"".join(map(str, numbers2))}"
    _file1 = os.path.join(RESNETS_FOLDER, f"{name1}.pt")
    _file2 = os.path.join(RESNETS_FOLDER, f"{name2}.pt")

    print("Loading Models and moving to Cuda")
    model1 = make_resnet(
        name1,
        BasicBlock,
        numbers1,
        False,
        False,
        num_classes=10,
    )
    model2 = make_resnet(
        name2,
        BasicBlock,
        numbers2,
        False,
        False,
        num_classes=10,
    )
    model1 = model1.cuda()
    model2 = model2.cuda()
    model1.load_state_dict(torch.load(_file1))
    model2.load_state_dict(torch.load(_file2))
    
    print("Getting loaders for FFCV")
    train_loader, test_loader = get_loaders(args)

    acc = evaluate(model, test_loader)
    print("Model Original Accuracy: {}".format(acc))

    # Everyone recieves, but not everyone sends
    # We do it this way for maximal simplicity
    print("Creating Tables, padding with None (and zero) to make it square")
    labels1 = ["input", "conv1"] + [(i, j) for j in range(0, len(numbers1[i])) for i in range(1,5)] + ["fc", "output"]
    labels2 = ["input", "conv1"] + [(i, j) for j in range(0, len(numbers2[i])) for i in range(1,5)] + ["fc", "output"]
    layerlabels = [
        [(labels1[i], labels2[j]) for j in range(len(labels2))] \
        for i in range(len(labels1))
    ]
    num_lables1 = num_labels(labels1)
    num_lables2 = num_labels(labels2)
    assert num_labels1 == len(labels1)
    assert num_labels2 == len(labels2)

    stitches = [
        [
            make_stitch(labels[0], labels[1]) \
            # Do not support stitching into input because nothing goes into input
            # Do not support stitching from output because output does not have an output
            # Do not support stitching from fc because matching shapes is hard
            if (labels[1] != "input" and labels[0] != "fc" and labels[0] != "output") else None \
            for labels in row
        ] \
        for row in layerlabels
    ]
    sims = [
        # 0.0 as default signifies "infinitely far away" or "no similarity"
        # because we couldn't stitch at all.
        [0.0 for _ in range(num_labels2)] \
        for _ in range(num_labels1)
    ]

    # Make sure all the lengths are correct
    assert len(layerlabels) == num_labels1
    assert len(stitches) == num_labels1
    assert len(sims) == num_labels1
    assert max((len(l) for l in layerlabels)) == num_labels2
    assert max((len(l) for l in stitches)) == num_labels2
    assert max((len(l) for l in sims)) == num_labels2
    assert min((len(l) for l in layerlabels)) == num_labels2
    assert min((len(l) for l in stitches)) == num_labels2
    assert min((len(l) for l in sims)) == num_labels2

    print("Training Table")
    for i in range(num_labels1):
        for j in range(num_labels2):
            # None is used to signify that this is not supported/stitchable
            if stitches[i][j]:
                print("*************************")
                ORIGINAL_PARAMS = pclone(model)
            
                send_label, recv_label = layerlabels[i][j]
                print(f"Training {send_label} to {recv_label}")
                stitch = stitches[i][j]
                stitch = stitch.cuda()
                print(stitch)
                stitched_resnet = make_stitched_resnet(model, stitch, send_label, recv_label)
                acc = train_loop(args, stitched_resnet, train_loader, test_loader)
                print(acc)
                sims[i][j] = acc

                NEW_PARAMS = pclone(model)
                assert listeq(ORIGINAL_PARAMS, NEW_PARAMS)
                print("*************************\n")


    print("Saving similarities")
    if not os.path.exists(SIMS_FOLDER):
        os.mkdir(SIMS_FOLDER)
    if not os.path.exists(HEATMAPS_FOLDER):
        os.mkdir(HEATMAPS_FOLDER)
    sim_path = os.path.join(SIMS_FOLDER, f"{name1}_{name2}_sims.pt")
    heat_path = os.path.join(HEATMAPS_FOLDER, f"{name1}_{name2}_heatmaps.png")
    torch.save(torch.tensor(sims), sim_path)
    matrix_heatmap(sim_path, heat_path, tick_labels_y=labels1, tick_labels_x=labels2)

class Args:
    def __init__(self):
        # FFCV Number of workers for loading
        self.fccv_num_workers: int = 1
        self.num_workers = 1

        # Used by FFCV for train/test split
        self.fraction = 1.0

        # Training Hyperparams
        self.bsz = 256   # Batch Size
        self.lr = 0.01   # Learning Rate
        self.warmup = 10  # Warmup epochs
        self.epochs = 7  # Total epochs
        self.wd = 0.01   # Weight decay
        self.dataset = "cifar10"

if __name__ == "__main__":
    # NOTE: 1111 -> ResNet10, 2222 -> ResNet18

    args = Args()
    number_pairs = [
        # Try Identity 1111 -> 1111
        ([1,1,1,1], [1,1,1,1]),
        # Try identity 2211 -> 2211 and 1122 -> 1122
        ([2,2,1,1], [2,2,1,1]),
        ([1,1,2,2], [1,1,2,2]),
        # Try 1111 -> 111 with a single 2 somewhere
        ([1,1,1,1], [1,1,1,2]),
        ([1,1,1,1], [1,1,2,1]),
        ([1,1,1,1], [1,2,1,1]),
        ([1,1,1,1], [2,1,1,1]),
        # Try 1111 -> increasing 2's from the right
        ([1,1,1,1], [1,1,2,2]),
        ([1,1,1,1], [1,2,2,2]),
        ([1,1,1,1], [2,2,2,2]),
        # Try 1111 -> increasing 2's from the left
        ([1,1,1,1], [2,2,1,1]),
        ([1,1,1,1], [2,2,2,1]),
        ([1,1,1,1], [2,2,2,2]),
        # Try the previous 2 experiments flipped
        ([1,1,2,2], [1,1,1,1]),
        ([1,2,2,2], [1,1,1,1]),
        ([2,2,2,2], [1,1,1,1]),
        # ...
        ([2,2,1,1], [1,1,1,1]),
        ([2,2,2,1], [1,1,1,1]),
        ([2,2,2,2], [1,1,1,1]),
    ]

    for pair in number_pairs:
        print(f"Training {pair[0]} to {pair[1]}")
        stitchtrain(pair[0], pair[1], args)
        # NOTE 2x long star to denote breakpoint between pairs
        print("**************************************************\n")