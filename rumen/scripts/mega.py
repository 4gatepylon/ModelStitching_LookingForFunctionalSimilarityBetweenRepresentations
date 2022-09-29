# A Massive script to try and do the entire stitching experiment (in its simplest possible form)
# without any external context.
import os
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


def get_loaders_no_ffcv(args):
    use_cuda = torch.cuda.is_available()
    assert use_cuda, "Should be using CUDA"
    train_batch_size = args.bsz
    test_batch_size = args.test_bsz

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
            adjust_learning_rate(epochs=epochs,
                                 warmup_epochs=args.warmup,
                                 base_lr=args.lr * args.bsz / 256,
                                 optimizer=optimizer,
                                 loader=train_loader,
                                 step=it)
            # zero grad (should we set to none?)
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

    print("Getting loaders NOT for FFCV, NO Shuffling")
    train_loader, test_loader = get_loaders_no_ffcv(args)

    # The assertions make sure that the dataset behaves deterministically
    print("Train and Test Accuracies are on the same dataset")
    acc1 = evaluate(model1, test_loader)
    acc1train = evaluate(model1, train_loader)
    print(f"Model 1 Test Accuracy: {acc1}, Train Accuracy: {acc1train}")
    acc2 = evaluate(model2, test_loader)
    acc2train = evaluate(model2, train_loader)
    print(f"Model 2 Test Accuracy: {acc2}, Train Accuracy: {acc2train}")

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
    # The two sims will be evaluated at different points
    sims_original = [
        # 0.0 as default signifies "infinitely far away" or "no similarity"
        # because we couldn't stitch at all.
        [0.0 for _ in range(num_labels)] \
        for _ in range(num_labels)
    ]
    sims_evaled = deepcopy(sims_original)
    sims_loaded = deepcopy(sims_original)

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

    print("Training Table (saving)")
    for i in range(num_labels):
        for j in range(num_labels):
            # None is used to signify that this is not supported/stitchable
            if stitches[i][j]:
                print("*************************")
                send_label, recv_label = layerlabels[i][j]
                stitch_file = os.path.join(STITCHES_FOLDER, f"stitch_{send_label}_{recv_label}.pt")

                print(f"Training {send_label} to {recv_label}")
                stitch = stitches[i][j]
                stitch = stitch.cuda()

                ORIGINAL_PARAMS_1 = pclone(model1)
                ORIGINAL_BUFFERS_1 = bclone(model1)
                ORIGINAL_PARAMS_2 = pclone(model2)
                ORIGINAL_BUFFERS_2 = bclone(model2)
                ORIGINAL_STITCH_PARAMS = pclone(stitch)
                ORIGINAL_STITCH_BUFFERS = bclone(stitch)
                

                stitched_resnet = make_stitched_resnet(model1, model2, stitch, send_label, recv_label)
                acc = train_loop(args, stitched_resnet, train_loader, test_loader)
                print(acc)
                sims_original[i][j] = acc

                # Save the stitch if this was a training run
                torch.save(stitch.state_dict(), stitch_file)

                NEW_PARAMS_1 = pclone(model1)
                NEW_BUFFERS_1 = bclone(model1)
                NEW_PARAMS_2 = pclone(model2)
                NEW_BUFFERS_2 = bclone(model2)
                NEW_STITCH_PARAMS = pclone(stitch)
                NEW_STITCH_BUFFERS = bclone(stitch)

                print("Asserting that stitch parameters changed, but not either model (train table)")
                assert listeq(ORIGINAL_PARAMS_1, NEW_PARAMS_1)
                assert listeq(ORIGINAL_BUFFERS_1, NEW_BUFFERS_1)
                assert listeq(ORIGINAL_PARAMS_2, NEW_PARAMS_2)
                assert listeq(ORIGINAL_BUFFERS_2, NEW_BUFFERS_2)
                assert not listeq(ORIGINAL_STITCH_PARAMS, NEW_STITCH_PARAMS)
                assert (len(ORIGINAL_STITCH_BUFFERS) == 0 and len(NEW_STITCH_BUFFERS) == 0) or \
                    not listeq(ORIGINAL_STITCH_BUFFERS, NEW_STITCH_BUFFERS)
                print("*************************\n")
    
    print("Creating Stitch Table by Evaluating")
    for i in range(num_labels):
        for j in range(num_labels):
            # None is used to signify that this is not supported/stitchable
            if stitches[i][j]:
                print("*************************")
                send_label, recv_label = layerlabels[i][j]

                print(f"Evaluating {send_label} to {recv_label}")
                stitch = stitches[i][j]
                stitch = stitch.cuda()

                ORIGINAL_PARAMS_1 = pclone(model1)
                ORIGINAL_BUFFERS_1 = bclone(model1)
                ORIGINAL_PARAMS_2 = pclone(model2)
                ORIGINAL_BUFFERS_2 = bclone(model2)
                ORIGINAL_STITCH_PARAMS = pclone(stitch)
                ORIGINAL_STITCH_BUFFERS = bclone(stitch)
                

                stitched_resnet = make_stitched_resnet(model1, model2, stitch, send_label, recv_label)
                acc = evaluate(stitched_resnet, test_loader)
                print(acc)
                sims_evaled[i][j] = acc

                # Save the stitch if this was a training run
                torch.save(stitch.state_dict(), stitch_file)

                NEW_PARAMS_1 = pclone(model1)
                NEW_BUFFERS_1 = bclone(model1)
                NEW_PARAMS_2 = pclone(model2)
                NEW_BUFFERS_2 = bclone(model2)
                NEW_STITCH_PARAMS = pclone(stitch)
                NEW_STITCH_BUFFERS = bclone(stitch)

                print("Asserting that no parameters changed after evaluation (eval table)")
                assert listeq(ORIGINAL_PARAMS_1, NEW_PARAMS_1)
                assert listeq(ORIGINAL_BUFFERS_1, NEW_BUFFERS_1)
                assert listeq(ORIGINAL_PARAMS_2, NEW_PARAMS_2)
                assert listeq(ORIGINAL_BUFFERS_2, NEW_BUFFERS_2)
                assert listeq(ORIGINAL_STITCH_PARAMS, NEW_STITCH_PARAMS)
                assert (len(ORIGINAL_STITCH_BUFFERS) == 0 and len(NEW_STITCH_BUFFERS) == 0) or \
                    listeq(ORIGINAL_STITCH_BUFFERS, NEW_STITCH_BUFFERS)
                print("*************************\n")
    
    print("Creating Stitch Table by Loading")
    for i in range(num_labels):
        for j in range(num_labels):
            # None is used to signify that this is not supported/stitchable
            if stitches[i][j]:
                print("*************************")
                send_label, recv_label = layerlabels[i][j]

                print(f"Loading {send_label} to {recv_label}")
                stitch = make_stitch(send_label, recv_label)
                stitch_file = os.path.join(STITCHES_FOLDER, f"stitch_{send_label}_{recv_label}.pt")
                stitch.load_state_dict(torch.load(stitch_file))
                stitch = stitch.cuda()

                true_stitch = stitches[i][j]
                true_stitch = true_stitch.cuda()

                TRUE_STITCH_PARAMS = pclone(true_stitch)
                TRUE_STITCH_BUFFERS = bclone(true_stitch)

                ORIGINAL_PARAMS_1 = pclone(model1)
                ORIGINAL_BUFFERS_1 = bclone(model1)
                ORIGINAL_PARAMS_2 = pclone(model2)
                ORIGINAL_BUFFERS_2 = bclone(model2)
                ORIGINAL_STITCH_PARAMS = pclone(stitch)
                ORIGINAL_STITCH_BUFFERS = bclone(stitch)

                print("Asserting that true stitch and loaded stitch have the same weights and buffers")
                assert(listeq(TRUE_STITCH_PARAMS, ORIGINAL_STITCH_PARAMS))
                assert(listeq(TRUE_STITCH_BUFFERS, ORIGINAL_STITCH_BUFFERS))

                stitched_resnet = make_stitched_resnet(model1, model2, stitch, send_label, recv_label)
                acc = evaluate(stitched_resnet, test_loader)
                print(acc)
                sims_loaded[i][j] = acc

                # Save the stitch if this was a training run
                torch.save(stitch.state_dict(), stitch_file)

                NEW_PARAMS_1 = pclone(model1)
                NEW_BUFFERS_1 = bclone(model1)
                NEW_PARAMS_2 = pclone(model2)
                NEW_BUFFERS_2 = bclone(model2)
                NEW_STITCH_PARAMS = pclone(stitch)
                NEW_STITCH_BUFFERS = bclone(stitch)

                # Stitch should change, but not the model
                print("Asserting that not parameters changed from loading evaluation")
                assert listeq(ORIGINAL_PARAMS_1, NEW_PARAMS_1)
                assert listeq(ORIGINAL_BUFFERS_1, NEW_BUFFERS_1)
                assert listeq(ORIGINAL_PARAMS_2, NEW_PARAMS_2)
                assert listeq(ORIGINAL_BUFFERS_2, NEW_BUFFERS_2)
                assert listeq(ORIGINAL_STITCH_PARAMS, NEW_STITCH_PARAMS)
                assert (len(ORIGINAL_STITCH_BUFFERS) == 0 and len(NEW_STITCH_BUFFERS) == 0) or \
                    listeq(ORIGINAL_STITCH_BUFFERS, NEW_STITCH_BUFFERS)
                print("*************************\n")
                

    print("Saving similarities")
    if not os.path.exists(SIMS_FOLDER):
        os.mkdir(SIMS_FOLDER)
    if not os.path.exists(HEATMAPS_FOLDER):
        os.mkdir(HEATMAPS_FOLDER)
    
    # Save the original, evaluated, and loaded tables
    # Models that are seperate are in different memory locations
    sepstr = "seperate" if models_seperate else "together"
    sims_original_path = os.path.join(SIMS_FOLDER, f"{name}_{name}_sims_original_{sepstr}_models.pt")
    sims_evaled_path = os.path.join(SIMS_FOLDER, f"{name}_{name}_sims_evaled_{sepstr}_models.pt")
    sims_loaded_path = os.path.join(SIMS_FOLDER, f"{name}_{name}_sims_loaded_{sepstr}_models.pt")
    heat_original_path = os.path.join(HEATMAPS_FOLDER, f"{name}_{name}_heatmaps_original_{sepstr}_models.png")
    heat_evaled_path = os.path.join(HEATMAPS_FOLDER, f"{name}_{name}_heatmaps_evaled_{sepstr}_models.png")
    heat_loaded_path = os.path.join(HEATMAPS_FOLDER, f"{name}_{name}_heatmaps_loaded_{sepstr}_models.png")

    torch.save(torch.tensor(sims_original), sims_original_path)
    torch.save(torch.tensor(sims_evaled), sims_evaled_path)
    torch.save(torch.tensor(sims_loaded), sims_loaded_path)

    matrix_heatmap(sims_original_path, heat_original_path, tick_labels_y=labels, tick_labels_x=labels)
    matrix_heatmap(sims_evaled_path, heat_evaled_path, tick_labels_y=labels, tick_labels_x=labels)
    matrix_heatmap(sims_loaded_path, heat_loaded_path, tick_labels_y=labels, tick_labels_x=labels)
    print("done!")

    print(f"Will Generate images in the images folder `{IMAGES_FOLDER}`")
    if not os.path.exists(IMAGES_FOLDER):
        os.mkdir(IMAGES_FOLDER)
    # 0th column is into input, 1st column is into conv1
    first_column_stitches = [
        stitches[i][1] \
        for i in range(len(stitches)) if stitches[i][1]
    ]
    send_labels = [
        layerlabels[i][1][0] \
        for i in range(len(layerlabels)) if stitches[i][1]
    ]
    first_column_images_folders = [
        os.path.join(IMAGES_FOLDER, f"row{i}_{sepstr}_models/") \
        for i in range(len(stitches)) if stitches[i][1]
    ]
    print("Making subfolder by row (sender)")
    for folder in first_column_images_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    num_image_pairs = 10
    assert len(first_column_stitches) == len(first_column_images_folders)
    assert len(send_labels) == len(first_column_images_folders)
    print(f"Generating {num_image_pairs} images")
    for st, folder, send_label in zip(first_column_stitches, first_column_images_folders, send_labels):
        # NOTE we use test_loader, but either should be fine
        # print(f"\tGenerating for Stitch {st}\n*************************\n")
        save_random_image_pairs(st, model1, send_label, num_image_pairs, folder, test_loader)

class Args:
    def __init__(self):
        # FFCV Number of workers for loading
        self.fccv_num_workers: int = 1
        self.num_workers = 1

        # Used by FFCV for train/test split
        self.fraction = 1.0

        # Training Hyperparams
        self.bsz = 256   # Batch Size
        self.test_bsz = 256
        self.lr = 0.01   # Learning Rate
        self.warmup = 10  # Warmup epochs
        # Total epochs per stitch to train (1 is good enough for our purposes)
        self.epochs = 1 # TODO
        self.wd = 0.01   # Weight decay
        self.dataset = "cifar10"


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