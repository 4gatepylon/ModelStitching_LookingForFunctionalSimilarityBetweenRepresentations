import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import math

from util import (
    kwargs,
    pool_output_dims,
    ensure_in_dict,
    ensure_not_in_dict,
    ensure_none,
    ensure_not_none,
)

# Layers are supposed to be the following:
# {
# (Required)      "layer_type" : "Conv2d" | "AvgPool2d" | "MaxPool2d" | "Linear" | "LogSoftmax" | "ReLU",
# (Conv and Pool) "kernel_size": an integer,
# (Conv)          "output_depth": an integer,
# (Conv and Pool) "stride": an integer,
# (Linear only)   "output_width": an integer,
# }
# NOTE: LogSoftmax and ReLU do not expect any parameters
# NOTE: we expect the stride of a pool to be the same as its kernel
# NOTE: expect no padding. Also, for regular pool behavior you should use a stride
# equal to the size of the kernel.
# This function will return layer, output_depth, output_height, output_width for convs and pools
# and just layer, width for linears
# @return [list of layers to append], {kwargs to next layers from layer dict}
# NOTE: while a lot of validity checking is done, you are still allowed to use LogSoftmax
# results in linear layers. This functionality is not meant to exist so ignore it.
def layers_from_layer_dict(layer, input_depth=None, input_height=None, input_width=None, mode_cnn=True):
    ensure_in_dict(layer, "layer_type")
    
    layer_type = layer["layer_type"]
    if layer_type == "Conv2d":
        if not mode_cnn:
            raise ValueError
        ensure_not_none(input_depth, input_height, input_width)
        ensure_in_dict(layer, "kernel_size", "output_depth", "stride")
        ensure_not_in_dict(layer, "output_widh")
        
        # NOTE: this was for debug, you can ignore it now
        # print(">>>>>>>>>> conv input dims {} x {}".format(input_height, input_width))

        # h, w = conv_output_dims(input_height, input_width, layer["kernel_size"], layer["stride"])
        # NOTE we only support (for now) zero-padding to offset the loss of length; we only support stride 1
        if layer["stride"] != 1:
            raise NotImplementedError("Stride was {} but we only support 1 for 1x1 convolution size matching (stitch)".format(
                layer["stride"]))
        pad = math.floor(layer["kernel_size"] / 2)
        h, w = input_height, input_width
        conv = nn.Conv2d(
            input_depth,
            layer["output_depth"], layer["kernel_size"], stride=layer["stride"],
            padding=pad, padding_mode="zeros")
        return [conv], kwargs(input_depth=layer["output_depth"], input_height=h, input_width=w, mode_cnn=True)
    
    # NOTE: we only support a single pool right after all the convolutions
    elif layer_type == "AvgPool2d" or layer_type == "MaxPool2d":
        if not mode_cnn:
            raise ValueError
        ensure_not_none(input_height, input_width)
        ensure_not_in_dict(layer, "output_width", "output_depth")
        ensure_in_dict(layer, "kernel_size", "stride")
        
        # NOTE: this for debug, you can ignore it now
        # print(">>>>>>>>>> pool input dims {} x {}".format(input_height, input_width))

        h, w = pool_output_dims(input_height, input_width, layer["kernel_size"], layer["stride"])
        pool = (nn.AvgPool2d if layer_type == "AvgPool2d" else nn.MaxPool2d)(layer["kernel_size"], stride=layer["stride"], padding=0)

        # NOTE: this was for debug you can ignore it now
        # print(">>>>>>>>>> pool output dims {} x {}".format(h, w))

        return [pool], kwargs(input_depth=input_depth, input_height=h, input_width=w, mode_cnn=True)
        
    elif layer_type == "Linear":
        ensure_not_in_dict(layer, "kernel_size", "output_depth", "stride")
        ensure_in_dict(layer, "output_width")
        
        # On layers that go from a convolution to an FC we need to flatten first
        if mode_cnn:
            ensure_not_none(input_depth, input_height, input_width)
        else:
            ensure_none(input_depth, input_height)
            ensure_not_none(input_width)

        layers = []
        input_width = input_width * input_height * input_depth if mode_cnn else input_width
        if mode_cnn:
            layers.append(Flattener())
        layers.append(nn.Linear(input_width, layer["output_width"]))
        
        return layers, kwargs(input_width=layer["output_width"], mode_cnn=False)
    
    elif layer_type == "ReLU":
        ensure_not_in_dict(layer, "kernel_size", "stride", "output_depth", "output_width")

        # We need to make sure to pass the right kwargs to the next one, though ReLU
        # can ALWAYS be applied to any tensor since it's pointwise
        k = kwargs(input_width=input_width, mode_cnn=False)
        if mode_cnn:
            ensure_not_none(input_depth, input_height, input_width)
            k = kwargs(input_depth=input_depth, input_height=input_height, input_width=input_width, mode_cnn=True)
        
        return [nn.ReLU(inplace=False)], k
    
    elif layer_type == "LogSoftmax":
        if mode_cnn:
            raise ValueError
        ensure_not_in_dict(layer, "kernel_size", "stride", "output_depth", "output_width")
        ensure_none(input_depth, input_height)
        
        return [nn.LogSoftmax(dim=1)], kwargs(input_width=input_width, mode_cnn=False)

    else:
        raise NotImplementedError

class Flattener(nn.Module):
    def __init__(self):
        super(Flattener, self).__init__()
    def forward(self, x):
        return torch.flatten(x, 1)

class Net(nn.Module):
    # This is a very simple convolutional network for 28x28x1 MNIST images:
    # 1. convolution
    #    ReLU
    #    ...
    # 2. flattener
    # 3. fc
    #    ...
    # 4. fc to classes (with ReLUs between)
    # 5. log softmax
    # NOTE:
    # Between any two convolutional layers there may be a pool.
    # Pools are either
    # NOTE:
    # - NO batch norm
    # - NO dropout
    # - NO regularization
    # - NO lack of symmetry: stride and convolutional width is the same in the x direction as it is in the y direction
    # - NO padding for average pools, and ALWAYS padding for convolutions (to maintain dimensions for stitching)
    # - NO dilation in convolutions or pools
    # - NO fancy pooling: only max pools and average pools with zero padding and symmetric strides
    # NOTE:
    # YOU are responsible for declaring ReLU and classification layers. YOU are responsible for knowing
    # what to stitch or not to stitch!
    def __init__(self, layers=None, input_shape=(1,28,28), mode_cnn=True):
        super(Net, self).__init__()
        if layers is None:
            raise NotImplementedError
        
        # Tells us which layers (by indices in the array) are pools
        # so that when we stitch we can stitch across only with layers at the same pool
        # (if we so desire)
        self.pools_idxs = []

        # Input width will be re-used for fully connected layers to be the width of the linear layer
        _input_depth, _input_height, _input_width = input_shape
        kwargs = {
            "mode_cnn": mode_cnn,
            "input_depth": _input_depth,
            "input_height": _input_height,
            "input_width": _input_width,
        }

        # Layers are just nn layers... you must already know where to stitch!
        if not layers:
            raise NotImplementedError
        self.layers = []
        for layer in layers:
            to_append, kwargs = layers_from_layer_dict(layer, **kwargs)
            for l in to_append:
                # NOTE: none of these are regularly registered since they are hidden in the list
                for pnum, p in enumerate(l.parameters()):
                    self.register_parameter(name="p{}-l{}".format(pnum, len(self.layers)), param=p)
                    # print("registered parameter p{}-l{}".format(pnum, len(self.layers)))
                self.layers.append(l)
        
    # Pass in the input to layer at input_idx and then continue passing on
    # to the output of the layer before output_idx (meaning that input is inclusive
    # and output is exclusive... i.e. the output_idx determines the index of the layer
    # to which the output you get would be going).
    def forward(self, x, input_idx=None, output_idx=None):
        if input_idx is None:
            input_idx = 0
        if output_idx is None:
            output_idx = len(self.layers)
        for idx in range(input_idx, output_idx):
            x = self.layers[idx](x)
        return x

# Stitch by putting an input in model 1 and taking the output from layer
# layer1 - 1 (i.e. the input that would go into layer 1) then passing that through a stitch
# and feeding that into layer2 of the second model and returning it.
# NOTE: compares layer1 - 1 and layer2
def stitch_forward(x, model1, model2, stitch, layer1_idx, layer2_idx):
    latent = model1(x, output_idx=layer1_idx)
    transformed = stitch(latent)
    y = model2(transformed, input_idx=layer2_idx)
    return y

class StitchedForward(nn.Module):
    def __init__(self, model1, model2, stitch, idx1, idx2):
        super(StitchedForward, self).__init__()
        self.m1 = model1
        self.m2 = model2
        self.i1 = idx1
        self.i2 = idx2
        self.s = stitch
    def forward(self, x):
        return stitch_forward(x, self.m1, self.m2, self.s, self.i1, self.i2)


# Given two layer indices find the stitch from the input of layer1 to the input
# if layer2 (i.e. get the shape that layer 1 is outputting and get what was output INTO
# it, then put that into layer 2)
# NOTE: gets the stitch to compare layer1 - 1 and layer2 - 1
def get_stitch(model1, model2, layer1_idx, layer2_idx, device):
    assert(type(model1) == Net)
    assert(type(model2) == Net)

    # We do not yet support stitching the first layer
    if layer1_idx == 0 or layer2_idx == 0:
        raise NotImplementedError

    # For these, if that was another waitless layer (of which the only allowable option is ReLU) then do so again
    layer1 = model1.layers[layer1_idx]
    layer2 = model2.layers[layer2_idx]
    dec_layer1 = False
    dec_layer2 = False

    if type(layer1) in [nn.ReLU, nn.MaxPool2d, nn.AvgPool2d, nn.LogSoftmax]:
        dec_layer1 = True
        layer1 = model1.layers[layer1_idx - 1]
    if type(layer1) == nn.ReLU:
        dec_layer1 = True
        layer1 = model1.layers[layer1_idx - 2]
    
    if type(layer2) in [nn.ReLU, nn.MaxPool2d, nn.AvgPool2d, nn.LogSoftmax]:
        dec_layer2 = True
        layer2 = model2.layers[layer2_idx - 1]
    if type(layer2) == nn.ReLU:
        dec_layer2 = True
        layer2 = model2.layers[layer2_idx - 2]

    if type(layer1) == nn.Linear and type(layer2) == nn.Linear:
        # For linear, since we pass vectors as colums the format becomes (output dim, input dim)
        # and we want to input the output dim of layer1 and output the output dim of layer2

        # Format by default is
        # OUTPUT, INPUT
        _, input_dim = layer1.weight.size()
        if dec_layer1:
            input_dim, _ = layer1.weight.size()
        _, output_dim = layer2.weight.size()
        if dec_layer2:
            output_dim, _ = layer2.weight.size()
        return nn.Linear(input_dim, output_dim).to(device)
    
    elif type(layer1) == nn.Conv2d and type(layer2) == nn.Conv2d:
        # Depending on whether we decrement or not, we want to use the output (of the previous layer)
        # or the input (of the current layer). Remember that the layer indices tell us what layer would
        # be recieving the output from network 1, and what layer WILL be recieving the output from network 2,
        # or if decremented what layer will be GIVING the size in network 1, and what size would be given
        # otherwise in network 2.

        # Format by default is
        # OUTPUT, INPUT, _, _ (so get input by default then if dec try output)
        _, input_depth, _, _ = layer1.weight.size()
        if dec_layer1:
            input_depth, _, _, _ = layer1.weight.size()
        _, output_depth, _, _ = layer2.weight.size()
        if dec_layer2:
            output_depth, _, _, _ = layer2.weight.size()
        
        # Stride is 1 because we are selecting which filters we want
        # and note that nothing else can change the width or height since
        # either there was a ReLU after this (does nothing to the dimensions)
        # or we will raise NotImplementedError
        return nn.Conv2d(input_depth, output_depth, 1, stride=1).to(device)
        
    else:
        # We do not support linear -> conv, conv-> linear, or any other fancy things
        raise NotImplementedError("Tried to stitch (dims from) idx {} to idx {} of types {} and {}".format(
            layer1_idx, layer2_idx, type(layer1), type(layer2)))

# Given two models and a dictionary of indices
# {index in model 1: [list of indices in model2 that we want to try to stitch with]}
# return a dictionary {index in model 1 : {index in model 2: stitch layer}}
# NOTE: idx1 should always be the layer we output from and idx2 should be the layer we input to
def get_stitches(model1, model2, idxs, device):
    return {idx1 : {idx2 : get_stitch(model1, model2, idx1, idx2, device) for idx2 in idx2s} for idx1, idx2s in idxs.items()}