import torch
import torch.nn as nn
import torch.nn.functional as F
########################################################################################################################


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
        ratio = 2**(blockSet - 1)
        snd_shape = (64 * ratio, 32 // ratio, 32 // ratio)

    if reciever == "conv1":
        return snd_shape, (3, 32, 32)
    elif reciever == "fc":
        return snd_shape, 512  # block expansion = 1 for resnet 18 and 34
    else:
        # NOTE we need to give the shape that the EXPECTED SENDER gives, NOT that of the reciever
        blockSet, block = reciever
        if block == 0:
            blockSet -= 1
        if blockSet == 0:
            # It's the conv1 expectee
            return snd_shape, (64, 32, 32)
        # ^^^
        ratio = 2**(blockSet - 1)
        rcv_shape = (64 * ratio, 32 // ratio, 32 // ratio)
    return snd_shape, rcv_shape
########################################################################################################################

########################################################################################################################


def resnet18_34_stitch(snd_shape, rcv_shape):
    if type(snd_shape) == int:
        raise Exception("can't send from linear layer")

    snd_depth, snd_hw, _ = snd_shape
    if type(rcv_shape) == int:
        # you can pass INTO an fc
        # , dtype=torch.float16))
        return nn.Sequential(nn.Flatten(), nn.Linear(snd_depth * snd_hw * snd_hw, rcv_shape))

    # else its tensor to tensor
    rcv_depth, rcv_hw, _ = rcv_shape
    upsample_ratio = rcv_hw // snd_hw
    downsample_ratio = snd_hw // rcv_hw

    # Downsampling (or same size: 1x1) is just a strided convolution since size decreases always by a power of 2
    # every set of blocks (blocks are broken up into sets that, within those sets, have the same size).
    if downsample_ratio >= upsample_ratio:
        # print(f"DOWNSAMPLE {snd_shape} -> {rcv_shape}: depth={snd_depth} -> {rcv_depth}, kernel_width={downsample_ratio}, stride={downsample_ratio}")
        # , dtype=torch.float16)
        return nn.Conv2d(snd_depth, rcv_depth, downsample_ratio, stride=downsample_ratio, bias=True)
    else:
        return nn.Sequential(
            nn.Upsample(scale_factor=upsample_ratio, mode='nearest'),
            nn.Conv2d(snd_depth, rcv_depth, 1, stride=1, bias=True))  # , dtype=torch.float16))
########################################################################################################################

########################################################################################################################
# This will create a layer to layer stitch table


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
            transformations[0][j] = resnet18_34_stitch(
                *resnet18_34_stitch_shape("conv1", into))
            idx2label[(0, j)] = ("conv1", into)
            j += 1
    # print(f"[0][{j}]: conv1 -> fc")
    transformations[0][j] = resnet18_34_stitch(
        *resnet18_34_stitch_shape("conv1", "fc"))
    idx2label[(0, j)] = ("conv1", "fc")
    transformations[0][0] = resnet18_34_stitch(
        *resnet18_34_stitch_shape("conv1", "conv1"))
    idx2label[(0, 0)] = ("conv1", "conv1")

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
                    transformations[i][j] = resnet18_34_stitch(
                        *resnet18_34_stitch_shape(outfrom, into))
                    idx2label[(i, j)] = (outfrom, into)
                    j += 1
            # print(f"[{i}][{j}]: {outfrom} -> fc")
            transformations[i][j] = resnet18_34_stitch(
                *resnet18_34_stitch_shape(outfrom, "fc"))
            idx2label[(i, j)] = (outfrom, "fc")
            transformations[i][0] = resnet18_34_stitch(
                *resnet18_34_stitch_shape(outfrom, "conv1"))
            idx2label[(i, 0)] = (outfrom, "conv1")
            j += 1
            i += 1
    # print(idx2label)
    return transformations, None, idx2label


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
            transformations[0][j] = resnet18_34_stitch(
                *resnet18_34_stitch_shape("conv1", into))
            idx2label[(0, j)] = ("conv1", into)
            j += 1
    # print(f"[0][{j}]: conv1 -> fc")
    transformations[0][j] = resnet18_34_stitch(
        *resnet18_34_stitch_shape("conv1", "fc"))
    idx2label[(0, j)] = ("conv1", "fc")
    transformations[0][0] = resnet18_34_stitch(
        *resnet18_34_stitch_shape("conv1", "conv1"))
    idx2label[(0, 0)] = ("conv1", "conv1")

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
                    transformations[i][j] = resnet18_34_stitch(
                        *resnet18_34_stitch_shape(outfrom, into))
                    idx2label[(i, j)] = (outfrom, into)
                    j += 1
            # print(f"[{i}][{j}]: {outfrom} -> fc")
            transformations[i][j] = resnet18_34_stitch(
                *resnet18_34_stitch_shape(outfrom, "fc"))
            idx2label[(i, j)] = (outfrom, "fc")
            transformations[i][0] = resnet18_34_stitch(
                *resnet18_34_stitch_shape(outfrom, "conv1"))
            idx2label[(i, 0)] = (outfrom, "conv1")
            j += 1
            i += 1
    return transformations, None, idx2label
########################################################################################################################

########################################################################################################################


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
########################################################################################################################
