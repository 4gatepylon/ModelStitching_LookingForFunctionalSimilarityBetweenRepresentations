# A Massive script to try and do the entire stitching experiment (in its simplest possible form)
# without any external context.
import os
import argparse
import math
import cv2
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
from mega_resnet import Resnet, BasicBlock, make_stitched_resnet, Identity

# NOTE same as the original because it's the same tree height away
RESNETS_FOLDER = "../../pretrained_resnets/"
SIMS_FOLDER = "../../sims/"
HEATMAPS_FOLDER = "../../heatmaps/"
STITCHES_FOLDER = "../../stitches/"
IMAGES_FOLDER = "../../images/"

# Loaders folders
NO_FFCV_FOLDER = "../../data_no_ffcv/"

NO_FFCV_CIFAR_MEAN = [0.1307, ]
NO_FFCV_CIFAR_STD = [0.3081, ]
INV_NO_FFCV_CIFAR_MEAN = [-ans for ans in NO_FFCV_CIFAR_MEAN]
INV_NO_FFCV_CIFAR_STD = [1.0/ans for ans in NO_FFCV_CIFAR_STD]

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

def bclone(model):
    return [p.data.detach().clone() for p in model.buffers()]


def listeq(l1, l2):
    if (len(l1) == 0 and len(l2) == 0):
        return True
    if (len(l1) != len(l2)):
        return False
    return min((torch.eq(a, b).int().min().item() for a, b in zip(l1, l2))) == 1

# Make sure that if you output from one layer and then input into the next one, 
# the following conditions will hold for a single network
# (and if they don't something is wrong):
# 1. You get equivalent accuracy
# 2. You get equivalent accuracy to the stitched network made this way
# NOTE this assumes Resnet1111
def sanity_test_diagonal(model1, model2, loader):
    labels = ["input", "conv1"] + [(i, 0) for i in range(1, 5)] + ["fc", "output"]
    assert len(labels) >= 5
    sane = True
    # For every pair of layers make sure that if you outut from one and input into the next one it works
    # Note we use warnings instead of asserts so that we can catch all the errors. This boolean function
    # is meant to be used in an assert so that -O can optimize it away for normal use.
    for i in range(0, len(labels) - 1):
        out_label = labels[i]
        in_label = labels[i+1]
        stitch = Identity()
        stitched_resnet = make_stitched_resnet(model1, model2, stitch, out_label, in_label)
        stitched_resnet.eval()
        for _input, _ in loader:
            with torch.no_grad():
                with autocast():
                    _input = _input.cuda()
                    _output = stitched_resnet(_input)
                    _intermediate_outfrom = model1.outfrom_forward(
                        _input,
                        out_label,
                    )
                    _output_into = model2.into_forward(
                        _intermediate_outfrom,
                        in_label,
                        pool_and_flatten=in_label == "fc",
                    )
                    _exp_output1 = model1(_input)
                    _exp_output2 = model2(_input)
                    # If the output is not equal to the expected output
                    exps_match_fail = (_exp_output1 == _exp_output2).int().min() != 1
                    stitched_network_fail = (_output == _exp_output1).int().min() != 1
                    outfrom_into_fail = (_output_into == _exp_output1).int().min() != 1
                    if exps_match_fail:
                        sane = False 
                        warn("Two models' expected outputs do not match")
                    if stitched_network_fail:
                        sane = False
                        warn(f"Stitched Resnet:")
                        warn(f"Sanity failed for {out_label} -> {in_label}")
                        warn(f"Got {_output}\n")
                        warn(f"Expected {_exp_output1}\n")
                        warn(f"Cutting off at first occurence for brevity")
                    if outfrom_into_fail:
                        sane = Fasle
                        warn(f"Using Outfrom & Into:")
                        warn(f"Sanity failed for {out_label} -> {in_label}")
                        warn(f"Got {_output}\n")
                        warn(f"Expected {_exp_output1}\n")
                        warn(f"Cutting off at first occurence for brevity")
                    if stitched_network_fail and not outfrom_into_fail:
                        warn(f"Stitched network failed, but out outfrom into, they might not be equal\n")
                    elif outfrom_into_fail and not stitched_network_fail:
                        warn(f"Outfrom/Into failed, but not the stitched network, they might not be equal")
    return sane


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


def get_loaders_no_ffcv(train_batch_size, test_batch_size):
    use_cuda = torch.cuda.is_available()
    assert use_cuda, "Should be using CUDA"

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': train_batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs =\
            {
                'num_workers': 0,
                # Pinning memory speeds up performance by copying
                # directly from disk to GPU I think.
                # https://spell.ml/blog/pytorch-training-tricks-YAnJqBEAACkARhgD
                'pin_memory': True
            }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    # Disable shuffling for determinism, but it shouldn't really matter that much
    # We keep it because it should (in theory, fingers crossed) help with accuracy
    train_kwargs['shuffle'] = True
    test_kwargs['shuffle'] = True

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            NO_FFCV_CIFAR_MEAN, NO_FFCV_CIFAR_STD)
    ])

    # NOTE the `train` parameter changes the actual data (I think) that is passed in (at least
    # it's coming from a different file). You can find it here
    # `https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10`
    # if you compare the test_list and train_list
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
    ### Hyperparameters ###
    bsz,
    wd,
    lr,
    max_epochs,
    warmup,
    ### ... ###
    model,
    train_loader,
    test_loader,
    early_stop_acc=None,
    verbose=True,
):
    parameters = list(model.parameters())

    assert bsz >= 256
    assert max_epochs >= 1

    optimizer = torch.optim.SGD(
        params=parameters,
        momentum=0.9,
        lr=lr * bsz / 256,
        weight_decay=wd
    )

    scaler = GradScaler()

    start = time.time()
    eval_acc = None
    for e in range(1, max_epochs + 1):
        # Log
        if verbose:
            print(f"\t\t starting on epoch {e} for {len(train_loader)} iterations")
        
        # Stop early if we have trained enough
        eval_acc = evaluate(model, test_loader)
        if (not early_stop_acc is None) and (early_stop_acc <= eval_acc):
            break
        
        model.train()
        # NOTE that enumerate's start changes the starting index
        for it, (inputs, outputs) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
            adjust_learning_rate(epochs=max_epochs,
                                 warmup_epochs=warmup,
                                 base_lr=lr * bsz / 256,
                                 optimizer=optimizer,
                                 loader=train_loader,
                                 step=it)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                y = outputs.cuda()
                h = inputs.cuda()
                h = model(h)
                loss = F.cross_entropy(h, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if verbose:
            print(f'\t\tepoch: {e} | time: {time.time() - start:.3f}')

    assert eval_acc > 0
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
        flat_shape = send_shape[0] * send_shape[1] * send_shape[2]
        return nn.Sequential(
            nn.Flatten(),
            # NOTE Adding BN because Yamini did it apparently? Unclear...
            nn.BatchNorm1d(flat_shape),
            nn.Linear(
                flat_shape,
                recv_shape,
            ),
        )
    else:
        send_depth, send_height, send_width = send_shape
        recv_depth, recv_height, recv_width = recv_shape
        if recv_height <= send_height:
            ratio = send_height // recv_height
            return nn.Sequential(
                # NOTE Adding BN because... refer to above
                # NOTE uses C from (N x C x H x W) according to 
                # https://pytorch.org/docs/master/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d
                nn.BatchNorm2d(send_depth),
                 nn.Conv2d(
                    send_depth,
                    recv_depth,
                    ratio,
                    stride=ratio,
                    bias=True,
                ),
                nn.BatchNorm2d(recv_depth),
            )
           
        else:
            ratio = recv_height // send_height
            return nn.Sequential(
                # NOTE Adding BN because... refer to above
                nn.BatchNorm2d(send_depth),
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
                nn.BatchNorm2d(recv_depth),
            )

def get_n_inputs(n, loader):
    k = 0
    for x, _ in loader:
        if k > n:
            break
        batch_size, _, _, _ = x.size()
        for i in range(min(batch_size, n - k)):
            # Output as a 4D tensor so that the network can take this as input
            y = x[i, :, :, :].flatten(end_dim=0).unflatten(0, (1, -1))
            yield y
        k += batch_size

def save_random_image_pairs(st, sender, send_label, num_pairs, foldername_images, train_loader):
    # NOTE st is not in cuda by default
    sender = sender.cuda()
    st = st.cuda()
    sender.eval()
    st.eval()
    for p in st.parameters():
        p.requires_grad = False
    
    # un-normalize the image so that it can be shown in cv2 how we would see it originally
    # NOTE this returns PIL images
    inv_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[0,], std=INV_NO_FFCV_CIFAR_STD),
        torchvision.transforms.Normalize(mean=INV_NO_FFCV_CIFAR_MEAN, std=[1,]),
        torchvision.transforms.ToPILImage(mode="RGB"),
    ])
    
    original_tensors = list(get_n_inputs(num_pairs, train_loader))
    for i in range(num_pairs):
        # Pick the filenames
        original_filename = os.path.join(
            foldername_images, f"original_{i}.png")
        generated_filename = os.path.join(
            foldername_images, f"generated_{i}.png")

        with torch.no_grad():
            with autocast():
                original_tensor = original_tensors[i].cuda()
                generated_tensor_pre = sender.outfrom_forward(
                    original_tensor,
                    send_label,
                )
                generated_tensor = st(generated_tensor_pre)

        # Save the images
        original_tensor_flat = original_tensor.flatten(end_dim=1)
        generated_tensor_flat = generated_tensor.flatten(end_dim=1)

        # NOTE these are PIL images
        original_tensor_flat = inv_transform(original_tensor_flat)
        generated_tensor_flat = inv_transform(generated_tensor_flat)
        original_np = np.array(original_tensor_flat)
        generated_np = np.array(generated_tensor_flat)

        assert original_np.shape == (32, 32, 3)
        assert generated_np.shape == (32, 32, 3)
        cv2.imwrite(original_filename, original_np)
        cv2.imwrite(generated_filename, generated_np)

def best_possible_stitch(models, labels, args, early_stop_acc = 0.9):
    assert type(models) == tuple and len(models) == 2
    assert type(labels) == tuple and len(labels) == 2
    assert not args.use_default_bsz
    assert not args.use_default_lr
    assert not args.use_default_max_epochs

    model1, model2 = models
    assert type(model1) == nn.Module
    assert type(model2) == nn.Module
    send_label, recv_label = labels
    assert type(send_label) == str or \
        (type(send_label) == tuple and len(send_label) == 2 and send_label[0] in [1, 2, 3, 4] and send_label[1] == 0)
    assert type(recv_label) == str or \
        (type(recv_label) == tuple and len(recv_label) == 2 and recv_label[0] in [1, 2, 3, 4] and recv_label[1] == 0)

    best_stitch = None
    best_acc = 0.0
    for max_epochs in args.max_epochs_range:
        for lr in args.lr_range:
            for batch_size in args.batch_size_range:
                train_loader, test_loader = get_loaders_no_ffcv(batch_size, args.test_bsz)

                stitch = make_stitch(send_label, recv_label)
                stitch = st.cuda()
                ORIGINAL_STITCH_PARAMS = pclone(stitch)
                ORIGINAL_STITCH_BUFFERS = bclone(stitch)

                
                stitched_resnet = make_stitched_resnet(model1, model2, stitch, send_label, recv_label)
                acc = train_loop(args, 
                    # Pass in our hyperparamters
                    batch_size,  # Changes
                    args.wd,     # Does NOT change
                    lr,          # Changes
                    max_epochs,  # Changes
                    warmup,      # Does NOT change
                    # ...
                    stitched_resnet,
                    train_loader,
                    test_loader,
                    early_stop_acc=early_stop_acc,
                    # Use Default Verbosity
                )

                # TODO add sanity test for file load/eval AND add hyperparameter search HERE
                # Save the stitch if this was a training run ONLY for the sanity test
                # torch.save(stitch.state_dict(), stitch_file)


                NEW_STITCH_PARAMS = pclone(stitch)
                NEW_STITCH_BUFFERS = bclone(stitch)
                assert not listeq(ORIGINAL_STITCH_PARAMS, NEW_STITCH_PARAMS)
                assert (len(ORIGINAL_STITCH_BUFFERS) == 0 and len(NEW_STITCH_BUFFERS) == 0) or \
                    not listeq(ORIGINAL_STITCH_BUFFERS, NEW_STITCH_BUFFERS)
                
    assert not best_stitch is None
    return (best_stitch, best_acc)
# NOTE that this code
# 1. does NOT pretrain (i.e. regular train) the models...
#    instead you can use `experiment.py` for that
# 2. does NOT use controls (i.e. random networks) since
#    we've more or less used them before and want to get
#    a quick sanity test batch of results


def stitchtrain(args, models_seperate=True):
    name = f"resnet_1111"
    numbers = [1, 1, 1, 1]
    _file = os.path.join(RESNETS_FOLDER, f"{name}.pt")

    print("Loading Models (one with init, one with DeepCopy) and moving to Cuda")
    model1 = Resnet(BasicBlock, numbers, num_classes=10)
    # NOTE we may want to just reference instead here...
    model2 = deepcopy(model1) if models_seperate else model1

    print("Asserting that a deepcopied model is the same as the original by weights (BEFORE loading)")
    assert(listeq(pclone(model1), pclone(model2)))
    print("Asserting that a deepcopied model is the same as the original by buffers (BEFORE loading)")
    assert(listeq(bclone(model1), bclone(model2)))

    model1 = model1.cuda()
    model2 = model2.cuda()
    model1.load_state_dict(torch.load(_file))
    model2.load_state_dict(torch.load(_file))

    print("Asserting that a deepcopied model is the same as the original by weights (AFTER loading)")
    assert(listeq(pclone(model1), pclone(model2)))
    print("Asserting that a deepcopied model is the same as the original by buffers (AFTER loading)")
    assert(listeq(bclone(model1), bclone(model2)))

    print("Disabling requires_grad and evaluating both models")
    for p in model1.parameters():
        p.requires_grad = False
    for p in model2.parameters():
        p.requires_grad = False
    model1.eval()
    model2.eval()

    print("Checking that parameters and buffers are the same for the models (AFTER evaluating)")
    assert listeq(list(model1.parameters()), list(model2.parameters())), \
        "The models need to have identical parameters"
    assert listeq(list(model1.buffers()), list(model2.buffers())), \
        "The models need to have identical buffers"
    assert(list(set(map(lambda p: p.data.dtype, model1.parameters()))) == [torch.float32]), \
        "The model1 needs to have only float32 parameters"
    assert(list(set(map(lambda p: p.data.dtype, model2.parameters()))) == [torch.float32]), \
        "The model2 needs to have only float32 parameters"

    print("Getting sanity loaders NOT for FFCV, NO Shuffling")
    train_loader, test_loader = get_loaders_no_ffcv(args.default_bsz, args.test_bsz)

    # The assertions make sure that the dataset behaves deterministically
    print("Train and Test Accuracies are on the same dataset")
    acc1 = evaluate(model1, test_loader)
    acc1train = evaluate(model1, train_loader)
    print(f"Model 1 Test Accuracy: {acc1}, Train Accuracy: {acc1train}")
    acc2 = evaluate(model2, test_loader)
    acc2train = evaluate(model2, train_loader)
    print(f"Model 2 Test Accuracy: {acc2}, Train Accuracy: {acc2train}")

    # NOTE this snippet is added to early stop training on easy layer pairs to stitch
    # when doing grid search
    max_acc = max(acc1, acc2)
    assert 0 <= max_acc and max_acc <= 1
    early_stop_acc = max_acc - (args.early_stop_percent_diff * max_acc)

    print("Making sure that the accuracies don't change")
    assert acc1 == evaluate(model1, test_loader), "Test Accuracy Shouldn't Change (1)"
    assert acc2 == evaluate(model2, test_loader), "Test Accuracy Shouldn't Change (2)"
    assert acc1train == evaluate(model1, train_loader), "Train Accuracy Shouldn't Change (1)"
    assert acc2train == evaluate(model2, train_loader), "Train Accuracy Shouldn't Change (2)"
    # NOTE test and train accuracy are NOT same because they come from DIFFERENT sub-datasets

    print("Sanity testing both models' \"Stitch Matrix\" Diagonals for correct output with Idenity stitch")
    assert sanity_test_diagonal(model1, model2, train_loader), "Sanity Test Diagonal (model1 -> model2) Failed on train_loader"
    assert sanity_test_diagonal(model1, model2, test_loader), "Sanity Test Diagonal (model1 -> model2) Failed on test_loader"

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
    # The two sims will be evaluated at different points
    sims_original = [
        # 0.0 as default signifies "infinitely far away" or "no similarity"
        # because we couldn't stitch at all.
        [0.0 for _ in range(num_labels)] \
        for _ in range(num_labels)
    ]

    # Make sure all the lengths are correct
    print("Ensuring that layerlabels, stitches, and sims_original tables have the proper dimensions")
    assert len(layerlabels) == num_labels
    assert len(stitches) == num_labels
    assert len(sims_original) == num_labels
    assert max((len(l) for l in layerlabels)) == num_labels
    assert max((len(l) for l in stitches)) == num_labels
    assert max((len(l) for l in sims_original)) == num_labels
    assert min((len(l) for l in layerlabels)) == num_labels
    assert min((len(l) for l in stitches)) == num_labels
    assert min((len(l) for l in sims_original)) == num_labels

    print("Ensuring that stitches folder exists")
    if not os.path.exists(STITCHES_FOLDER):
        os.mkdir(STITCHES_FOLDER)
    
    print(f"Will Generate images in the images folder `{IMAGES_FOLDER}`")
    if not os.path.exists(IMAGES_FOLDER):
        os.mkdir(IMAGES_FOLDER)

    # `sepstr` is used in various places below to annotate what experiment this is
    # (we run both with one model and with two just in case as a debugging tactic)
    sepstr = "seperate" if models_seperate else "together"

    # NOTE train loader is not used and new ones are created for the purposes of hyperparameter tuning
    train_loader = None
    print("Training Table (saving)")
    for i in range(num_labels):
        for j in range(num_labels):
            # None is used to signify that this is not supported/stitchable
            if stitches[i][j]:
                print("*************************")
                send_label, recv_label = layerlabels[i][j]
                stitch_file = os.path.join(STITCHES_FOLDER, f"stitch_{send_label}_{recv_label}.pt")

                print(f"Training {send_label} to {recv_label}")
                # NOTE we do this sanity test wrapping the grid search (on each entry)
                # because it will decrease the amount of work done and we have confidence that
                # there are no bugs right now
                ORIGINAL_PARAMS_1 = pclone(model1)
                ORIGINAL_BUFFERS_1 = bclone(model1)
                ORIGINAL_PARAMS_2 = pclone(model2)
                ORIGINAL_BUFFERS_2 = bclone(model2)

                # NOTE stitch will be in cuda
                stitch, acc = best_possible_stitch(
                    models=(model1, model2),
                    labels=(send_label, recv_label),
                    args=args,
                    # NOTE this may make more sense to define somewhere else, but whatever
                    early_stop_acc=early_stop_acc,
                )
                print(acc)

                NEW_PARAMS_1 = pclone(model1)
                NEW_BUFFERS_1 = bclone(model1)
                NEW_PARAMS_2 = pclone(model2)
                NEW_BUFFERS_2 = bclone(model2)
                
                print("Asserting that model parameters did not change after grid search")
                assert listeq(ORIGINAL_PARAMS_1, NEW_PARAMS_1)
                assert listeq(ORIGINAL_BUFFERS_1, NEW_BUFFERS_1)
                assert listeq(ORIGINAL_PARAMS_2, NEW_PARAMS_2)
                assert listeq(ORIGINAL_BUFFERS_2, NEW_BUFFERS_2)
                print("*************************\n")

                # j == 1 is the same as recv_label == "conv1"
                if j == 1:
                    assert recv_label == "conv1"
                    print(f"Generating image for stitch {send_label} -> {recv_label}")
                    img_folder = os.path.join(IMAGES_FOLDER, f"row{i}_{sepstr}_models/")
                    print(f"Storing image in {img_folder}")
                    if not os.path.exists(img_folder):
                        os.mkdir(img_folder)
                    num_image_pairs = 10
                    save_random_image_pairs(stitch, model1, send_label, num_image_pairs, img_folder, test_loader)

    print("Creating similarities and heatmaps folder")
    if not os.path.exists(SIMS_FOLDER):
        os.mkdir(SIMS_FOLDER)
    if not os.path.exists(HEATMAPS_FOLDER):
        os.mkdir(HEATMAPS_FOLDER)
    
    print("Saving similarities and heatmaps")
    sims_original_path = os.path.join(SIMS_FOLDER, f"{name}_{name}_sims_original_{sepstr}_models.pt")
    heat_original_path = os.path.join(HEATMAPS_FOLDER, f"{name}_{name}_heatmaps_original_{sepstr}_models.png")
    torch.save(torch.tensor(sims_original), sims_original_path)
    matrix_heatmap(sims_original_path, heat_original_path, tick_labels_y=labels, tick_labels_x=labels)
    print("done!")
        

class Args:
    def __init__(self):        
        ### Batch sizes ###
        self.use_default_bsz = False             # ...
        self.default_bsz = 1024                  # ...
        self.bsz_range = [256, 1024, 2048, 4096] # Grid search on this, NOTE 256 is min because of lr_scheduler (read the code)
        self.test_bsz = 256                      # NOTE test batch size shouldn't affect acc

        ### Learning Rate(s) ###
        self.use_default_lr = False                   # ...
        self.default_lr = 0.01                        # ...
        self.lr_range = [0.001, 0.01, 0.05, 0.1, 0.2] # Grid Search on this

        ### Epochs ###
        self.warmup = 10                          # Warmup epochs
        self.use_default_max_epochs = False       # ...
        self.default_max_epochs = 1               # ...
        self.max_epochs_range = [1, 5, 8, 10, 15] # Grid search on this
        self.use_early_stop = True                # ...
        self.early_stop_percent_diff = 0.01       # If you get within this * max original model acc, stop training
        
        ### Misc ###
        self.wd = 0.01           # Weight decay
        self.dataset = "cifar10" # ...


if __name__ == "__main__":
    # I get this runtime error if I don't do this:
    # Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or
    # `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because
    # it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must
    # set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8
    #or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to
    # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    fix_seed(0)

    args = Args()
    # NOTE it should not matter whether models are seperate or not, the answer should
    # be the exact same.
    print(f"Models Seperate\n**************************************************\n")
    stitchtrain(args, models_seperate=True)
    print(f"\n**************************************************\n")
    print(f"Models Together\n**************************************************\n")
    stitchtrain(args, models_seperate=False)
    print(f"\n**************************************************\n")

    # Cleaar this in case we want to run faster again later
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG")