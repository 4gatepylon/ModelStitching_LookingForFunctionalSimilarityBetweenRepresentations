# A Massive script to try and do the entire stitching experiment (in its simplest possible form)
# without any external context.
import os
import math
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

from copy import deepcopy
from mega_resnet import Resnet, BasicBlock

# NOTE same as the original because it's the same tree height away
RESNETS_FOLDER = "../../pretrained_resnets/"
SIMS_FOLDER = "../../sims/"
HEATMAPS_FOLDER = "../../heatmaps/"
STITCHES_FOLDER = "../../stitches/"

# Loaders folders
NO_FFCV_FOLDER = "../../data_no_ffcv/"

NO_FFCV_CIFAR_MEAN = [0.1307, ]
NO_FFCV_CIFAR_STD = [0.3081, ]


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Suggested from
    # https://discuss.pytorch.org/t/evaluate-twice-accuracy-changes-if-i-shuffle/150976
    torch.use_deterministic_algorithms(True)


def pclone(model):
    return [p.data.detach().clone() for p in model.parameters()]


def listeq(l1, l2):
    return min((torch.eq(a, b).int().min().item() for a, b in zip(l1, l2))) == 1


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
    test_batch_size = args.test_bsz

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': train_batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs =\
            {
                # Single worker for determinism
                'num_workers': 1,
                # Pinning memory speeds up performance by copying
                # directly from disk to GPU I think.
                # https://spell.ml/blog/pytorch-training-tricks-YAnJqBEAACkARhgD
                'pin_memory': True
            }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    # Disable shuffling for determinism
    train_kwargs['shuffle'] = False
    test_kwargs['shuffle'] = True

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            NO_FFCV_CIFAR_MEAN, NO_FFCV_CIFAR_STD)
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
    model.eval()

    total_correct, total_num = 0., 0.
    for _, (images, labels) in enumerate(test_loader):
        with torch.no_grad():
            labels = labels.cuda().double()
            img = images.cuda()
            h = model(img)
            preds = h.argmax(dim=1).double()
            total_correct += (preds == labels).sum().cpu().item()
            total_num += h.shape[0]

    return total_correct / total_num


def train_loop(
    args,
    model,
    train_loader,
    test_loader,
    verbose=True,
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
            print(
                f"\t\t starting on epoch {e} for {len(train_loader)} iterations")
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
                # NOTE this is for ffcv
                y = outputs.cuda()
                h = inputs.cuda()
                h = model(h)
                # print(h)
                # print(y)
                # TODO modularize this out to enable sim training
                loss = F.cross_entropy(h, y)
                # print(loss)

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


def stitchtrain(args, two_models=False, load_stitch=False):
    # I get this runtime error if I don't do this:
    # Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or
    # `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because
    # it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must
    # set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8
    #or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to
    # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    fix_seed(0)

    name = f"resnet_1111"
    numbers = [1, 1, 1, 1]
    _file = os.path.join(RESNETS_FOLDER, f"{name}.pt")

    print("Loading Models and moving to Cuda")
    model1 = Resnet(BasicBlock, numbers, num_classes=10)
    model2 = model1
    if two_models:
        # NOTE normally we might load instead as in the commented section
        model2 = deepcopy(model1)
        # model2 = Resnet(BasicBlock, numbers, num_classes=10)

    model1 = model1.cuda()
    model2 = model2.cuda()
    model1.load_state_dict(torch.load(_file))
    model2.load_state_dict(torch.load(_file))

    print("Disabling requires_grad and evaluating both models")
    for p in model1.parameters():
        p.requires_grad = False
    for p in model2.parameters():
        p.requires_grad = False
    model1.eval()
    model2.eval()

    print("Checking that parameters and buffers are the same for the models")
    assert listeq(list(model1.parameters()), list(model2.parameters())), \
        "The models need to have identical parameters"
    assert listeq(list(model1.buffers()), list(model2.buffers())), \
        "The models need to have identical buffers"
    assert(list(set(map(lambda p: p.data.dtype, model1.parameters()))) == [torch.float32]), \
        "The model1 needs to have only float32 parameters"
    assert(list(set(map(lambda p: p.data.dtype, model2.parameters()))) == [torch.float32]), \
        "The model2 needs to have only float32 parameters"

    print("Getting loaders NOT for FFCV, NO Shuffling")
    train_loader, test_loader = get_loaders_no_ffcv(args)

    # The assertions make sure that the dataset behaves deterministically
    acc1 = evaluate(model1, test_loader)
    print("Model 1 Original Accuracy: {}".format(acc1))
    acc2 = evaluate(model2, test_loader)
    print("Model 2 Original Accuracy: {}".format(acc2))

    print("Making sure that the accuracies don't change")
    assert acc1 == evaluate(model1, test_loader)
    assert acc2 == evaluate(model2, test_loader)

    print("Creating Tables, padding with None (and zero) to make it square")
    labels = ["input", "conv1"] + [(i, 0)
                                   for i in range(1, 5)] + ["fc", "output"]
    layerlabels = [
        [(labels[i], labels[j]) for j in range(len(labels))]
        for i in range(len(labels))
    ]
    num_labels = 8
    assert len(labels) == 8
    assert len(layerlabels) == 8
    assert max(map(len, layerlabels)) == 8 and min(map(len, layerlabels)) == 8

    print("Creating stitches and sims")
    # 1x1 or 2x2 table of stitch table
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
        [0.0 for _ in range(num_labels)] \
        for _ in range(num_labels)
    ]

    # Make sure all the lengths are correct
    print("Ensuring that layerlabels, stitches, and sims tables have the proper dimensions")
    assert len(layerlabels) == num_labels
    assert len(stitches) == num_labels
    assert len(sims) == num_labels
    assert max((len(l) for l in layerlabels)) == num_labels
    assert max((len(l) for l in stitches)) == num_labels
    assert max((len(l) for l in sims)) == num_labels
    assert min((len(l) for l in layerlabels)) == num_labels
    assert min((len(l) for l in stitches)) == num_labels
    assert min((len(l) for l in sims)) == num_labels

    print("Ensuring that stitches folder exists")
    if not os.path.exists(STITCHES_FOLDER):
        os.mkdir(STITCHES_FOLDER)

    # # TODO
    # # stitched resnet's forward is:
    # # def forward(self, x):
    # # h = self.sender.outfrom_forward(
    # #     x,
    # #     self.send_label,
    # # )
    # # h = self.stitch(h)
    # # h = self.reciever.into_forward(
    # #     h,
    # #     self.recv_label,
    # #     pool_and_flatten=self.recv_label == "fc",
    # # )
    # # return h
    # print("Training Table")
    # for i in range(num_labels):
    #     for j in range(num_labels):
    #         # None is used to signify that this is not supported/stitchable
    #         if stitches[i][j]:
    #             print("*************************")
    #             ORIGINAL_PARAMS_1 = pclone(model1)
    #             ORIGINAL_PARAMS_2 = pclone(model2)

    #             send_label, recv_label = layerlabels[i][j]
    #             stitch_file = os.path.join(
    #                 STITCHES_FOLDER, f"stitch_{send_label}_{recv_label}.pt")

    #             print(f"Training {send_label} to {recv_label}")
    #             stitch = stitches[i][j]
    #             # If we already have a stitch, just load it and evaluate it for the acc
    #             # Else retrain it (noting that train loop does evaluate at the end)
    #             if load_stitch:
    #                 stitch.load_state_dict(torch.load(stitch_file))
    #             # ...
    #             stitch = stitch.cuda()

    #             stitched_resnet = make_stitched_resnet(
    #                 model1, model2, stitch, send_label, recv_label)
    #             acc = evaluate(stitched_resnet, test_loader) \
    #                 if load_stitch else \
    #                 train_loop(args, stitched_resnet,
    #                            train_loader, test_loader)
    #             # ...
    #             print(acc)
    #             sims[i][j] = acc

    #             # Save the stitch if this was a training run
    #             if not load_stitch:
    #                 torch.save(stitch.state_dict(), stitch_file)

    #             NEW_PARAMS_1 = pclone(model1)
    #             NEW_PARAMS_2 = pclone(model2)
    #             assert listeq(ORIGINAL_PARAMS_1, NEW_PARAMS_1)
    #             assert listeq(ORIGINAL_PARAMS_2, NEW_PARAMS_2)
    #             print("*************************\n")

    # print("Saving similarities")
    # if not os.path.exists(SIMS_FOLDER):
    #     os.mkdir(SIMS_FOLDER)
    # if not os.path.exists(HEATMAPS_FOLDER):
    #     os.mkdir(HEATMAPS_FOLDER)
    # sim_path = os.path.join(
    #     SIMS_FOLDER, f"{name}_{name}_sims_load_stitch_{load_stitch}_two_models_{two_models}.pt")
    # heat_path = os.path.join(
    #     HEATMAPS_FOLDER, f"{name}_{name}_heatmaps_load_stitch_{load_stitch}_two_models_{two_models}.png")
    # torch.save(torch.tensor(sims), sim_path)
    # matrix_heatmap(sim_path, heat_path, tick_labels_y=labels,
    #                tick_labels_x=labels)
    print("done!")

    # Cleaar this in case we want to run faster again later
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG")


class Args:
    def __init__(self):
        # FFCV Number of workers for loading
        self.fccv_num_workers: int = 1
        self.num_workers = 1

        # Used by FFCV for train/test split
        self.fraction = 1.0

        # Training Hyperparams
        self.bsz = 256   # Batch Size
        self.test_bsz = 2048
        self.lr = 0.01   # Learning Rate
        self.warmup = 10  # Warmup epochs
        # Total epochs per stitch to train (1 is good enough for our purposes)
        self.epochs = 1
        self.wd = 0.01   # Weight decay
        self.dataset = "cifar10"


if __name__ == "__main__":
    args = Args()
    stitchtrain(args, two_models=True, load_stitch=False)
    # # Create regular stitches (one model) then load them once into
    # # a single model (should give a 2nd diagonal) and then into a
    # # pair of models (unknown what will happen)
    # stitchtrain(args, two_models=False, load_stitch=False)
    # stitchtrain(args, two_models=False, load_stitch=True)
    # stitchtrain(args, two_models=True, load_stitch=True)
    # # Retrain and confirm that we get a triangle (i.e. code isn't buggy)
    # stitchtrain(args, two_models=True, load_stitch=False)
