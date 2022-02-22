###
### NOTE this copied from stitching-public with small modifications
###
from warnings import warn

import torch.nn as nn
import torch.nn.functional as F

# For more information look at
# https://www.kaggle.com/greatcodes/pytorch-cnn-resnet18-cifar10
# http://pytorch.org/vision/master/_modules/torchvision/models/resnet.html#resnet18

## ResNet18 for CIFAR
## Based on: https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
class PreActBlock(nn.Module):
    # Pre-activation version of the BasicBlock.
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class View(nn.Module):
    FlattenView = lambda out: (out.size(0), -1)
    # Expects a function that given an x will return the shape to view it as
    def __init__(self, viewshape):
        self.viewshape = viewshape
        super(View, self).__init__()
    def forward(self, x):
        return x.view(*list(self.viewshape(x)))

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, init_channels=64):
        super(PreActResNet, self).__init__()
        self.in_planes = init_channels
        c = init_channels

        # We begin with a convolutional layer to get the right shape
        self.conv1 = nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=False)

        # Then we have 4 residual-style block layers
        self.layer1 = self._make_layer(block, c, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*c, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*c, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*c, num_blocks[3], stride=2)

        # Classifier will be last layer
        self.avg_pool = nn.AvgPool2d(4)
        self.flattener = View(View.FlattenView)
        self.linear = nn.Linear(8*c*block.expansion, num_classes)

        self.classifier = nn.Sequential(self.avg_pool, self.flattener, self.linear)

        # This sequence makes it easy to insert intermediate inputs and take out intermediate outputs
        self.layers = [self.conv1, self.layer1, self.layer2, self.layer3, self.layer4, self.classifier]

    def _make_layer(self, block, planes, num_blocks, stride):
        # eg: [2, 1, 1, ..., 1]. Only the first one downsamples.
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # Modified to enable stitching; in_layer is the layer that we are inputting into
    # while outfrom layer is the layer we are pulling the output from. We assume 5 layers total.
    def forward(self, x, into_layer=0, outfrom_layer=5):
        assert into_layer <= outfrom_layer
        assert 0 <= into_layer and into_layer < len(self.layers)
        assert 0 <= outfrom_layer and outfrom_layer < len(self.layers)

        out = x
        for idx in range(into_layer, outfrom_layer+1):
            layer = self.layers[idx]
            out = layer(out)
        return out

# This creates a resnet stitch
# we are willing to stitch from, into
# 0 -> 0, 1, 2, 3, 4, 5 (simply change depth and downsample, or flatten and project)
# 1 -> 1, 2, 3, 4, 5    (simply change depth and downsample or flatten and project)
# 2 -> 1, 2, 3, 4, 5    (simply change depth and downsample, or flatten and project)
# 3 -> 1, 2, 3, 4, 5    (simply change depth and downsample, or flatten and project)
# 4 -> 1, 2, 3, 4, 5    (simply change depth and downsample, or flatten and project)
# 5 -> nothing
#
# And note that stitching
# i -> j
# is the same as comparing i with j - 1
#
# The parameter format is:
# out_layer: the layer that is output from in the first network
# in_layer: the layer that we input into in the second network
# k: the number of channels that we told each network (should be the same) to start with
# img_height: the height of the image
# img_width: the width of the image
# NOTE: it will be worth exploring whether we want bias or not...
def create_resnet_stitch(out_layer, in_layer, k=64, width_divider=2, img_height=32, img_width=32, img_chans=3, linear_classes=10):
    # Ensure that the stitch is 
    if k != 64:
        warn("If you pick k={} which is not 64, your stitch may not be supported".format(k))
    if width_divider != 2:
        warn("If you pick width_divider={} which is not 2, your stitch may not be supported".format(width_divider))
    if img_width != img_height:
        warn("Non-square images may not be supported, you input img_height={}, img_width={}".format(img_height, img_width))
    if img_height != 32 or img_width != 32:
        warn("Image is not cifar height/width: img_height={}, img_width={}".format(img_height, img_width))
    if linear_classes != 10:
        warn("Targetting {} classes which is not 10 may not be supported".format(linear_classes))
    if img_chans != 3:
        warn("Trying to stitch with {} channels, may not be supported".format(img_chans))

    # We support only three stitches:
    # Layer 0 to any layer,
    # Layer 1,2,3,4 to any layer AFTER it
    #
    # Anything else will not work
    out_is_block = 1 <= out_layer and out_layer <= 4
    in_is_block = 1 <= in_layer and in_layer <= 4
    if in_is_block and out_is_block:
        # We will expect to get something with height and width
        # out_hw and depth out_depth
        out_hw = img_height / (2**(out_layer-1))
        out_depth = k * 2**(out_layer-1)

        # We will want to return something with in_hw height, width
        # and in_depth depth
        in_hw = img_height / (2**(in_layer-1))
        in_depth = k * 2**(in_layer-1)
        ratio = in_hw / out_hw
        if in_hw > out_hw:
            # If we need to upsample, we a form of upsampling that basically copies it
            # https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
            return nn.Sequential(
                nn.Upsample(scale_factor=ratio, mode='nearest'),
                nn.Conv2d(out_depth, in_depth, 1, stride=1, bias=True))
        else:
            # If we need to downsample, we just use a nxn convolution with an n-size stride due to the
            # shapes of the resnets (documented below).
            assert ratio % 2 == 0
            assert 1 <= ratio and ratio <= 8
            return nn.Conv2d(out_depth, in_depth, ratio, stride=ratio, bias=True)
        
    elif out_layer == 0:
        initial_hw_dim = 32
        in_depth = k
        if in_is_block:
            out_depth 
        elif in_layer == 5:
            in_dim = img_width * img_height * k
            out_dim = linear_classes
            # NOTE: we enable bias because self.conv1 doesn't have it
            return nn.Sequential(View(View.FlattenView), nn.Linear(in_dim, out_dim, bias=True))
        else:
            raise NotImplemented("in_layer={} is not supported for out_layer=0".format(in_layer))
    else:
        raise NotImplementedError("out_layer={} is not supported".format(out_layer))
    raise NotImplementedError

class StitchedResnet(nn.Module):
    def __init__(self, net1, net2, out_layer, in_layer):
        super(StitchedResnet, self).__init__()
        self.out_net = net1
        self.in_net = net2
        self.out_layer = out_layer
        self.in_layer = in_layer
        # We still pass in the defaults in case they are changed
        self.stitch = create_resnet_stitch(self.out_layer, self.in_layer, k=64, width_divider=2, img_height=32, img_width=32, img_chans=3, linear_classes=10)

    def forward(self, x):
        out = self.out_net(x, into_layer=0, outfrom_layer=self.out_layer)
        out = self.stitch(out)

        # Assume that there are a total of 6 layers (zero indexxed)
        out = self.in_net(out, into_layer=self.in_layer, outfrom_layer=5)
        return out
    
    def get_stitch(self):
        return self.stitch
    def get_model1(self):
        return self.out_net
    def get_model2(self):
        return self.in_net
    def get_layer1(self):
        return self.out_layer
    def get_layer2(self):
        return self.in_layer

RESNET18_SELF_STITCHES_COMPARE_FMT = {
    # The format here is outfrom, into
    # so the output from layer 0 can go into layer 1 as input (comparing zero with zero)
    0: [1, 2, 3, 4, 5],
    1: [1, 2, 3, 4, 5],
    2: [1, 2, 3, 4, 5],
    3: [1, 2, 3, 4, 5],
    4: [1, 2, 3, 4, 5],
    5: [],
}

# NOTE that we use compare format here
# This means that we must put in the layers we want to compare, which automatically
# corresponds to l1 -> stitche -> l2+1.
def stitched_resnet(net1, net2, l1, l2, mode="compare"):
    if mode != "compare":
        raise NotImplementedError("Only compare mode is supported. Please put in the pair of models and layers you want to compare")
    # Read the documentation right above create_resnet_stitch
    assert 0 <= l1 and l1 < 5
    assert 0 < l2 if 0 < l1 else True
    return StitchedResnet(net1, net2, l1, l2+1)

# This will return an nxn array of stitched nets for the valid layers
# which are one to one with the indices (and layers 1-4 inclusive are for blocks)
# There will be None where a stitch was not possible
# NOTE that we kn
def stitches_resnet(net1, net2, valid_stitches=RESNET18_SELF_STITCHES_COMPARE_FMT, in_mode="outfrom_into", out_mode="compare", min_layer=0, max_layer=5, device="cpu"):
    # Modes are
    # outfrom_into      (layer that outputs, layer that inputs)              = (sender, reciever)
    # compare | outfrom (layer that ouputs, layer that would have outputted) = (sender, expected sender)
    # into              (layer that would have inputted, layer that inputs)  = (expected reciever, reciever)

    if in_mode != "outfrom_into":
        raise NotImplementedError("You must pass in a dictionary of stitches {l1 : [l2, ...]} where l1 is the layer that outputs into the stitch, and l2 is the layer that takes the input from the stitch")
    if out_mode != "compare":
        raise NotImplementedError("Only compare mode is supported.")
    if min_layer != 0:
        raise NotImplementedError("We expect the minimum layer to be zero")
    
    N = max_layer - min_layer
    stitched_nets = [[None for _ in range(N)] for _ in range(N)]
    for outfrom, layers in valid_stitches.items():
        for into in layers:
            compare1 = outfrom
            compare2 = into - 1
            # You must compare layers that exist
            assert min_layer <= compare1 and compare1 <= max_layer
            assert min_layer <= compare2 and compare2 <= max_layer

            # Row is which is sender, 
            stitched_nets[compare1][compare2] = stitched_resnet(net1, net2, compare1, compare2, mode="compare").to(device)
    return stitched_nets

def resnet18k_cifar(k = 64, num_classes=10) -> PreActResNet: # k = 64 is the standard ResNet18
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes, init_channels=k)



### NOTE resnet stitch derivation/documentation:
# We will be trying to stitch the block output shapes
# We see that they are:
# [batch, k, 32, 32]
# [batch, k*2, 16, 16]
# [batch, k*4, 8, 8]
# [batch, k*8, 4, 4]
#
# Which for k=64 (which is fine for our purposes, we just need to get better optimization hyperparams)
# and ignoring batch size:
# 1: 64, 32, 32
# 2: 128, 16, 16
# 3: 256, 8, 8
# 4: 512, 4, 4
# which means that we simply need to do a 2x upsampling per to enable stitching how we want...
# we can use https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
# with either nearest or bilinear... I think that if we use nearest it should be possible
# to learn using convolutions... I'll start by trying nearest

# 1 -> 1 (same): nn 1x1 conv (64 -> 64), stride 1
# 1 -> 2 (down): nn 2x2 conv (64 -> 128), stride 2
# 1 -> 3 (down): nn 4x4 conv (128 -> 256), stride 4
# 1 -> 4 (down): nn 8x8 conv (256 -> 512), stride 8

# 2 -> 1 (up):   F upsample (scale_factor 2, mode nearest) -> 1x1 conv (128 -> 64), stride 1
# 2 -> 2 (same): nn 1x1 conv (128 -> 128), stride 1
# 2 -> 3 (down): nn 2x2 conv (128 -> 256), stride 2
# 2 -> 4 (down): nn 4x4 conv (128 -> 512), stride 4

# 3 -> 1 (up):   F upsample (scale_factor 4, mode nearest) -> 1x1 conv (256 -> 64), stride 1
# 3 -> 2 (up):   F upsample (scale_factor 2, mode nearest) -> 1x1 conv (256 -> 128), stride 1
# 3 -> 3 (same): nn 1x1 conv (256 -> 256), stride 1
# 3 -> 4 (down): nn 2x2 conv (256 -> 512), stride 2

# 4 -> 1 (up):   F upsample (scale_factor 8, mode nearest) -> 1x1 conv (128 -> 64), stride 1
# 4 -> 2 (up):   F upsample (scale_factor 4, mode nearest) -> 1x1 conv (512 -> 128), stride 1
# 4 -> 3 (up):   F upsample (scale_factor 2, mode nearest) -> 1x1 conv (512 -> 256), stride 1
# 4 -> 4 (same): nn 1x1 conv (512 -> 512), stride 1

# How might we generalize this?
  
# Sometimes torchsummary fails for some reason (i.e. I get some wierd libiomp5.dylib error)
# So here is the torch summary pasted for your convenience for a resnet18 for cifar w/
# batch size 512:
#
# PreActResNet                             --                        --
# ├─Conv2d: 1-1                            [512, 64, 32, 32]         1,728
# ├─Sequential: 1-2                        [512, 64, 32, 32]         --
# │    └─PreActBlock: 2-1                  [512, 64, 32, 32]         --
# │    │    └─BatchNorm2d: 3-1             [512, 64, 32, 32]         128
# │    │    └─Conv2d: 3-2                  [512, 64, 32, 32]         36,864
# │    │    └─BatchNorm2d: 3-3             [512, 64, 32, 32]         128
# │    │    └─Conv2d: 3-4                  [512, 64, 32, 32]         36,864
# │    └─PreActBlock: 2-2                  [512, 64, 32, 32]         --
# │    │    └─BatchNorm2d: 3-5             [512, 64, 32, 32]         128
# │    │    └─Conv2d: 3-6                  [512, 64, 32, 32]         36,864
# │    │    └─BatchNorm2d: 3-7             [512, 64, 32, 32]         128
# │    │    └─Conv2d: 3-8                  [512, 64, 32, 32]         36,864
# ├─Sequential: 1-3                        [512, 128, 16, 16]        --
# │    └─PreActBlock: 2-3                  [512, 128, 16, 16]        --
# │    │    └─BatchNorm2d: 3-9             [512, 64, 32, 32]         128
# │    │    └─Sequential: 3-10             [512, 128, 16, 16]        8,192
# │    │    └─Conv2d: 3-11                 [512, 128, 16, 16]        73,728
# │    │    └─BatchNorm2d: 3-12            [512, 128, 16, 16]        256
# │    │    └─Conv2d: 3-13                 [512, 128, 16, 16]        147,456
# │    └─PreActBlock: 2-4                  [512, 128, 16, 16]        --
# │    │    └─BatchNorm2d: 3-14            [512, 128, 16, 16]        256
# │    │    └─Conv2d: 3-15                 [512, 128, 16, 16]        147,456
# │    │    └─BatchNorm2d: 3-16            [512, 128, 16, 16]        256
# │    │    └─Conv2d: 3-17                 [512, 128, 16, 16]        147,456
# ├─Sequential: 1-4                        [512, 256, 8, 8]          --
# │    └─PreActBlock: 2-5                  [512, 256, 8, 8]          --
# │    │    └─BatchNorm2d: 3-18            [512, 128, 16, 16]        256
# │    │    └─Sequential: 3-19             [512, 256, 8, 8]          32,768
# │    │    └─Conv2d: 3-20                 [512, 256, 8, 8]          294,912
# │    │    └─BatchNorm2d: 3-21            [512, 256, 8, 8]          512
# │    │    └─Conv2d: 3-22                 [512, 256, 8, 8]          589,824
# │    └─PreActBlock: 2-6                  [512, 256, 8, 8]          --
# │    │    └─BatchNorm2d: 3-23            [512, 256, 8, 8]          512
# │    │    └─Conv2d: 3-24                 [512, 256, 8, 8]          589,824
# │    │    └─BatchNorm2d: 3-25            [512, 256, 8, 8]          512
# │    │    └─Conv2d: 3-26                 [512, 256, 8, 8]          589,824
# ├─Sequential: 1-5                        [512, 512, 4, 4]          --
# │    └─PreActBlock: 2-7                  [512, 512, 4, 4]          --
# │    │    └─BatchNorm2d: 3-27            [512, 256, 8, 8]          512
# │    │    └─Sequential: 3-28             [512, 512, 4, 4]          131,072
# │    │    └─Conv2d: 3-29                 [512, 512, 4, 4]          1,179,648
# │    │    └─BatchNorm2d: 3-30            [512, 512, 4, 4]          1,024
# │    │    └─Conv2d: 3-31                 [512, 512, 4, 4]          2,359,296
# │    └─PreActBlock: 2-8                  [512, 512, 4, 4]          --
# │    │    └─BatchNorm2d: 3-32            [512, 512, 4, 4]          1,024
# │    │    └─Conv2d: 3-33                 [512, 512, 4, 4]          2,359,296
# │    │    └─BatchNorm2d: 3-34            [512, 512, 4, 4]          1,024
# │    │    └─Conv2d: 3-35                 [512, 512, 4, 4]          2,359,296
# ├─Linear: 1-6                            [512, 10]                 5,130

