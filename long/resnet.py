###
### NOTE this copied from stitching-public with small modifications
###

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
    # Expects a function that given an x will return the shape to view it as
    def __init__(self, viewshape):
        self.viewshape = viewshape
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
        self.flattener = View(lambda out: (out.size(0), -1))
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

# def resnet_stitch(in_shape, out_shape):
#     pass

# def stitched_resnet():
#     pass

def resnet18k_cifar(k = 64, num_classes=10) -> PreActResNet: # k = 64 is the standard ResNet18
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes, init_channels=k)
