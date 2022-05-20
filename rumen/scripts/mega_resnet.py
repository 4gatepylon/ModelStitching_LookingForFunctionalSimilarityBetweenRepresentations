# To avoid toooo much bloat we put all the resnet logic in its own file

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Resnet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            num_classes: int = 10,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation = None,
            norm_layer = None,
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
        # NOTE that default stride for _make_layer is 1 and default dilation is False
        # NOTE that _make_layer has side effects
        depths = [64, 128, 256, 512]
        strides = [1, 2, 2, 2]
        dilations = [False, replace_stride_with_dilation[0], replace_stride_with_dilation[1], replace_stride_with_dilation[2]]

        self.blocksets = nn.Sequential(*[
            self._make_layer(block, depths[i], layers[i], stride=strides[i], dilate=dilations[i])\
            for i in range(4)
        ])
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
                if isinstance(m, BasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)
                else:
                    raise NotImplementedError

    def _make_layer(
            self,
            block,
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

    # TODO vent must be refactored
    def into_forward(self, x, vent, pool_and_flatten = False):
        if type(vent) == str and vent == "input":
            raise Exception(
                "can vent out from input but NOT into input in `intoForward(x, vent)`")
        elif type(vent) == str and vent == "conv1":
            return self.forward(x)
        elif type(vent) == tuple:
            # Resnets vent every time so that we can shorten our code
            blockset, block = vent
            assert blockset >= 1
            assert blockset <= 4
            assert block < 4
            assert block >= 0
            assert len(self.blocksets) == 4

            if block < len(self.blocksets[blockset - 1]):
                x = self.blocksets[blockset - 1][block:](x)
            if blockset < len(self.blocksets):
                x = self.blocksets[blockset:](x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        elif type(vent) == str and vent == "fc":
            if pool_and_flatten:
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
            return self.fc(x)
        elif type(vent) == str and vent == "output":
            return x
        else:
            raise ValueError(f"Not input, conv1, fc, or block: {vent}")

    # TODO outfrom forward must be refactored
    def outfrom_forward(self, x, vent, pool_and_flatten = False):
        if pool_and_flatten:
            raise NotImplementedError
        
        # Special case to deal with the fact that sometimes we need to apply maxpool
        if type(vent) == str and vent == "conv1":
            # We make the choice to do bn1 and relu because the into_forward
            # from above does not do them. This is probably the better behavior
            # because otherwise it would be equivalent to stitching from the
            # input (image space).
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            return x
        elif type(vent) == str and vent == "fc":
            return self.forward(x)
        elif type(vent) == str and vent == "output":
            raise ValueError(f"Can't have output for `outfrom_forward`")
        elif type(vent) == str and vent == "input":
            return x
        
        assert type(vent) == tuple
        blockset, block = vent

        # When you want to flatten it in th end (and avgpool) you first do
        # the rest of the network, then you apply those transformations.
        if blockset == 4 and block + 1 == self.blocksets[-1] and pool_and_flatten:
            x = self.outfrom_forward(x, vent, pool_and_flatten=False)
            x = torch.flatten(self.avgpool(x), 1)
            return x

        # Normal case
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # NOTE we always take vent+1 because lists are exclusive in python
        assert blockset >= 1
        assert blockset <= 4
        assert block >= 0
        assert block < 4
        y = self.blocksets[:blockset-1](x)
        # assert bool((y == x if blockset == 1 else y != x).int().min().item())
        z = self.blocksets[blockset-1][:block + 1](y)
        return z

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)   # CIFAR10 modification

        x = self.blocksets(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# TODO
# Remove this if we keep getting wierd ass behavior
class StitchedResnet(nn.Module):
    def __init__(self, sender, reciever, stitch, send_label, recv_label):
        super(StitchedResnet, self).__init__()
        self.sender = sender
        self.reciever = reciever
        self.send_label = send_label
        self.recv_label = recv_label
        self.stitch = stitch

        self.sender.eval()
        self.reciever.eval()
        for p in self.sender.parameters():
            p.requires_grad = False
        
        for p in self.reciever.parameters():
            p.requires_grad = False
    
    # NOTE important that it return the stitch's parameters because
    # it will be used as a black box in the train loop.
    def parameters(self):
        return self.stitch.parameters()
    
    def train(self):
        self.stitch.train()
        # NOTE very important that sender and reciever are always evaled
        self.sender.eval()
        self.reciever.eval()
    
    def eval(self):
        self.stitch.eval()
        self.sender.eval()
        self.reciever.eval()

    def forward(self, x):
        h = self.sender.outfrom_forward(
            x,
            self.send_label,
        )
        h = self.stitch(h)
        h = self.reciever.into_forward(
            h, 
            self.recv_label,
            pool_and_flatten=self.recv_label == "fc",
        )
        return h

def make_stitched_resnet(model1, model2, stitch, send_label, recv_label):
    return StitchedResnet(model1, model2, stitch, send_label, recv_label)
# end
# TODO

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x