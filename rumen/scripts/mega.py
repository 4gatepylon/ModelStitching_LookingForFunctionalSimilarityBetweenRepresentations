# A Massive script to try and do the entire stitching experiment (in its simplest possible form)
# without any external context.

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from mega_resnet import BasicBlock
from mega_resnet import make_resnet
from mega_resnet import make_stitched_resnet

# NOTE same as the original because it's the same tree height away
RESNETS_FOLDER = "../../pretrained_resnets/"
SIMS_FOLDER = "../../sims/"
HEATMAPS_FOLDER = "../../heatmaps/"
STITCHES_FOLDER = "../../stitches/"

# Loaders folders
# FFCV_FOLDER = "../../data_ffcv/"
NO_FFCV_FOLDER = "../../data_no_ffcv/"
# MISC_FFCV_FOLDER = "../tmp/"

# FFCV_CIFAR_MEAN = [125.307, 122.961, 113.8575]
# FFCV_CIFAR_STD = [51.5865, 50.847, 51.255]
NO_FFCV_CIFAR_MEAN = [0.1307, ]
NO_FFCV_CIFAR_STD = [0.3081, ]
# FFCV_NORMALIZE_TRANSFORM = torchvision.transforms.Normalize(
#     FFCV_CIFAR_MEAN, FFCV_CIFAR_STD)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pclone(model):
    return [p.data.detach().clone() for p in model.parameters()]


def listeq(l1, l2):
    return min((torch.eq(a, b).int().min().item() for a, b in zip(l1, l2))) == 1

def num_labels(numbers):
    # input, conv1, ... one for each block, blockset, ... fc, output
    return 2 + 2 + sum(numbers)

# NOTE: tick labels for y before x because y is the sender
def matrix_heatmap(input_file_name: str, output_file_name: str, tick_labels_y=None, tick_labels_x=None):
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
        if tick_labels_x:
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

def get_loaders_no_ffcv(args):
        use_cuda = torch.cuda.is_available()
        train_batch_size = args.bsz
        test_batch_size = 2048

        device = torch.device("cuda" if use_cuda else "cpu")

        train_kwargs = {'batch_size': train_batch_size}
        test_kwargs = {'batch_size': test_batch_size}
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)
        train_kwargs['shuffle'] = True
        test_kwargs['shuffle'] = True

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(NO_FFCV_CIFAR_MEAN, NO_FFCV_CIFAR_STD)
        ])

        dataset1 = datasets.CIFAR10(
            NO_FFCV_FOLDER, train=True, download=True, transform=transform)
        dataset2 = datasets.CIFAR10(
            NO_FFCV_FOLDER, train=False, transform=transform)

        train_loader = DataLoader(dataset1, **train_kwargs)
        test_loader = DataLoader(dataset2, **test_kwargs)

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

    total_correct, total_num = 0., 0.
    for _, (images, labels) in enumerate(test_loader):

        with torch.no_grad():
            with autocast():
                # NOTE cuda is for not ffcv
                labels = labels.cuda()
                h = images.cuda()
                h = model(h)
                preds = h.argmax(dim=1)
                total_correct += (preds == labels).sum().cpu().item()
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
        for it, (inputs, outputs) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
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
                # NOTE this is for not ffcv
                y = outputs.cuda()
                h = inputs.cuda()
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
def stitchtrain(args):
    # NOTE an experiment
    fix_seed(0)

    name = f"resnet_1111"
    numbers = [1, 1, 1, 1]
    _file = os.path.join(RESNETS_FOLDER, f"{name}.pt")

    print("Loading Models and moving to Cuda")
    model1 = make_resnet(
        name,
        BasicBlock,
        numbers,
        False,
        False,
        num_classes=10,
    )
    model2 = model1
    # model2 = make_resnet(
    #     name,
    #     BasicBlock,
    #     numbers,
    #     False,
    #     False,
    #     num_classes=10,
    # )
    
    model1 = model1.cuda()
    # model2 = model2.cuda()
    model1.load_state_dict(torch.load(_file))
    # model2.load_state_dict(torch.load(_file))
    
    
    print("Getting loaders NOT FFCV")
    train_loader, test_loader = get_loaders_no_ffcv(args)
    
    # Why is the accuracy not the same?
    test_loader = train_loader
    acc1 = evaluate(model1, test_loader)
    acc2 = evaluate(model1, test_loader)
    print("Model 1 Original Accuracy: {}".format(acc1))
    print("Model 2 Original Accuracy: {}".format(acc2))
    print("Model 3 Stitched Accuracy: {}".format(evaluate(model1, test_loader)))
    print("Model 4 Stitched Accuracy: {}".format(evaluate(model1, test_loader)))
    print("Model 5 Stitched Accuracy: {}".format(evaluate(model1, test_loader)))
    print("Model 6 Stitched Accuracy: {}".format(evaluate(model1, test_loader)))

    assert acc1 == acc2 and False

    print("Creating Tables, padding with None (and zero) to make it square")
    labels = ["input", "conv1"] + [(i, 0) for i in range(1,5)] + ["fc", "output"]
    layerlabels = [
        [(labels[i], labels[j]) for j in range(len(labels))] \
        for i in range(len(labels))
    ]
    num_labels = 8
    assert len(labels) == 8
    assert len(layerlabels) == 8
    assert max(map(len, layerlabels)) == 8 and min(map(len, layerlabels)) == 8

    stitches_orig = [
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

    # For when we load weights
    stitches_loaded = [
        [
            make_stitch(labels[0], labels[1]) \
            if (labels[1] != "input" and labels[0] != "fc" and labels[0] != "output") else None \
            for labels in row
        ] \
        for row in layerlabels
    ]

    sims_orig = [
        # 0.0 as default signifies "infinitely far away" or "no similarity"
        # because we couldn't stitch at all.
        [0.0 for _ in range(num_labels)] \
        for _ in range(num_labels)
    ]

    # Similarities for when you re-load the weights
    sims_loaded = [
        [0.0 for _ in range(num_labels)] \
        for _ in range(num_labels)
    ]


    # Make sure all the lengths are correct
    assert len(layerlabels) == num_labels
    assert len(stitches_orig) == num_labels
    assert len(sims_orig) == num_labels
    assert max((len(l) for l in layerlabels)) == num_labels
    assert max((len(l) for l in stitches_orig)) == num_labels
    assert max((len(l) for l in sims_orig)) == num_labels
    assert min((len(l) for l in layerlabels)) == num_labels
    assert min((len(l) for l in stitches_orig)) == num_labels
    assert min((len(l) for l in sims_orig)) == num_labels

    print("Training Table")
    for i in range(num_labels):
        for j in range(num_labels):
            # None is used to signify that this is not supported/stitchable
            if stitches_orig[i][j]:
                print("*************************")
                ORIGINAL_PARAMS_1 = pclone(model1)
                ORIGINAL_PARAMS_2 = pclone(model2)
            
                send_label, recv_label = layerlabels[i][j]
                stitch_file = os.path.join(STITCHES_FOLDER, f"stitch_{send_label}_{recv_label}.pt")

                print(f"Training {send_label} to {recv_label}")
                stitch = stitches_orig[i][j]
                stitch = stitch.cuda()

                STITCHES_PARAMS = pclone(stitch)

                stitched_resnet = make_stitched_resnet(model1, model2, stitch, send_label, recv_label)
                acc = train_loop(args, stitched_resnet, train_loader, test_loader)
                
                print(acc)
                sims_orig[i][j] = acc

                # Save the stitch if this was a training run
                torch.save(stitch.state_dict(), stitch_file)

                NEW_PARAMS_1 = pclone(model1)
                NEW_PARAMS_2 = pclone(model2)
                NEW_STICHES_PARAMS = pclone(stitch)
                assert listeq(ORIGINAL_PARAMS_1, NEW_PARAMS_1)
                assert listeq(ORIGINAL_PARAMS_2, NEW_PARAMS_2)
                assert not listeq(STITCHES_PARAMS, NEW_STICHES_PARAMS)
                print("*************************\n")

    print("Evaluating Table From Loaded Stitch")
    for i in range(num_labels):
        for j in range(num_labels):
            if stitches_loaded[i][j]:
                print("*************************")
                ORIGINAL_PARAMS_1 = pclone(model1)
                ORIGINAL_PARAMS_2 = pclone(model2)
                
                send_label, recv_label = layerlabels[i][j]
                stitch_file = os.path.join(STITCHES_FOLDER, f"stitch_{send_label}_{recv_label}.pt")

                print(f"Evaluating and Loading {send_label} to {recv_label}")
                stitch = stitches_loaded[i][j]
                stitch = stitch.cuda()

                STITCH_ORIG_PARAMS = pclone(stitch)

                stitch.load_state_dict(torch.load(stitch_file))

                STITCH_LOADED_PARAMS = pclone(stitches_loaded[i][j])
                # TODO For some reason this doesn't work
                # assert listeq(STITCH_ORIG_PARAMS, STITCH_LOADED_PARAMS)

                acc = evaluate(stitch, test_loader)
                sims_loaded[i][j] = acc
                print(acc)
                print("*************************\n")


    print("Saving similarities")
    if not os.path.exists(SIMS_FOLDER):
        os.mkdir(SIMS_FOLDER)
    if not os.path.exists(HEATMAPS_FOLDER):
        os.mkdir(HEATMAPS_FOLDER)
    sim_path_orig = os.path.join(SIMS_FOLDER, f"{name}_{name}_sims_orig.pt") 
    sim_path_loaded = os.path.join(SIMS_FOLDER, f"{name}_{name}_sims_loaded.pt")
    heat_path_orig = os.path.join(HEATMAPS_FOLDER, f"{name}_{name}_heatmaps_orig.png")
    heat_path_loaded = os.path.join(HEATMAPS_FOLDER, f"{name}_{name}_heatmaps_loaded.png")
    torch.save(torch.tensor(sims_orig), sim_path_orig)
    torch.save(torch.tensor(sims_loaded), sim_path_loaded)
    # NOTE these two should look the exact same!
    matrix_heatmap(sim_path_orig, heat_path_orig, tick_labels_y=labels, tick_labels_x=labels)
    matrix_heatmap(sim_path_loaded, heat_path_loaded, tick_labels_y=labels, tick_labels_x=labels)

class Args:
    def __init__(self):
        # Training Hyperparams
        self.bsz = 256   # Batch Size
        self.lr = 0.01   # Learning Rate
        self.warmup = 10  # Warmup epochs
        self.epochs = 1  # Total epochs
        self.wd = 0.01   # Weight decay
        self.dataset = "cifar10"

if __name__ == "__main__":
    args = Args()
    stitchtrain(args)