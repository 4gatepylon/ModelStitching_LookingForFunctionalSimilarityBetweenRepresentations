# TODO this shit needs to be refactored
# TODO this shit needs to be refactored
# TODO this shit needs to be refactored
# TODO this shit needs to be refactored
# - prefix/suffix
# - stitched resnet
# - numerical shit

# modified from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
from typing import Type, Any, Callable, Union, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from layer_label import LayerLabel
from resnet.resnet_utils import (
    conv1x1,
)
from resnet.block import (
    BasicBlock,
    Bottleneck,
)


class Resnet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 10,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        # NOTE we need to finetune this when we load because the pretrained versions only have
        # the 7x7 imagenet convolutions
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # CIFAR10 modification
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # TODO add state to keep track of
        # - whether to flatten going out
        # - whether to go in or go out (vent into vs out from)
        self.blocksets = nn.Sequential(
            self.layer1, self.layer2, self.layer3, self.layer4)
        assert len(self.blocksets) == 4, \
            "The number of blocksets should be 4, got {}".format(
                len(self.blocksets))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # NOTE we add this back for stitching
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,  # This is layers[i] if this is the [i]th size layers
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def into_forward(self, x: Tensor, vent: LayerLabel, pool_and_flatten: bool = False) -> Tensor:
        if vent.isInput():
            raise Exception(
                "can vent out from input but NOT into input in `intoForward(x, vent)`")
        elif vent.isConv1():
            return self.forward(x)
        elif vent.isBlock():
            # Resnets vent every time so that we can shorten our code
            blockset, block = vent.getBlockset(), vent.getBlock()
            xp = self.blocksets[blockset-1][block:](x)
            if blockset < LayerLabel.BLOCKSET_MAX:
                xp = self.blocksets[blockset:](xp)
            a = self.avgpool(xp)
            f = torch.flatten(a, 1)
            y = self.fc(f)
            return y
        elif vent.isFc():
            if pool_and_flatten:
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
            return self.fc(x)
        elif vent.isOutput():
            return x
        else:
            raise ValueError(f"Not input, conv1, fc, or block: {vent}")

    def outfrom_forward(self, x: Tensor, vent: LayerLabel, pool_and_flatten: bool = False):
        # Special case to deal with the fact that sometimes we need to apply maxpool
        if vent.isConv1():
            # Are you sure we don't want to use the RELU? NOTE
            return self.conv1(x)
        elif vent.isFc():
            return self.forward(x)
        elif vent.isOutput():
            raise ValueError(f"Can't have output for `outfrom_forward`")
        elif vent.isInput():
            return x

        blockset, block = vent.getBlockset(), vent.getBlock()

        # When you want to flatten it in th end (and avgpool) you first do
        # the rest of the network, then you apply those transformations.
        if blockset == 4 and block + 1 == len(self.layer4) and pool_and_flatten:
            x = self.outfrom_forward(x, vent, pool_and_flatten=False)
            x = torch.flatten(self.avgpool(x), 1)
            return x

        # Normal case
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # NOTE we always take vent+1 because lists are exclusive in python
        x = self.blocksets[:blockset-1](x)
        y = self.blocksets[blockset-1][:block + 1](x)
        return y

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)   # CIFAR10 modification

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
