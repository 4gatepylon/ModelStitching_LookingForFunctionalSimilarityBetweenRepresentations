import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# These are two hard-coded examples with 3 and 10 convolutions respectively
# of sizes hard-coded by me. They both have 2 FCs followed by an FC classifier
# of the same shape/size
from examples_2 import NET_3_2, NET_10_2

# Calculate the width and height of a convolutional layer
# from the input width and height as well as the (assumed symmetric)
# width and stride. Assume no padding. Same for pools.
def conv_output_dims(height, width, kernel, stride):
    # We take the floor since there is no padding
    output_height = math.floor((height - kernel + 1) / stride)
    output_width = math.floor((width - kernel + 1) / stride)
    return output_height, output_width
def pool_output_dims(height, width, kernel, stride):
    # It's the same due to lack of padding
    return conv_output_dims(height, width, kernel, stride)

def kwargs(**kwargs):
    return kwargs

def ensure_in_dict(dictionary, *args):
    for arg in args:
        if not arg in dictionary:
            raise ValueError("Dictionary {} missing {}".format(dictionary, arg))
def ensure_not_in_dict(dictionary, *args):
    for arg in args:
        if arg in dictionary:
            raise ValueError("Dictionary {} should not have {}".format(dictionary, arg))

def ensure_not_none(*args):
    for arg in args:
        if arg is None:
            raise ValueError("Argument was None")
def ensure_none(*args):
    for arg in args:
        if not arg is None:
            raise validity("Argument was not None")


# Layers are supposed to be the following:
# {
# (Required)      "layer_type" : "Conv2d" | "AvgPool2d" | "MaxPool2d" | "Linear" | "LogSoftmax" | "ReLU",
# (Conv and Pool) "kernel_size": an integer,
# (Conv)          "output_depth": an integer,
# (Conv and Pool) "stride": an integer,
# (Linear only)   "output_width": an integer,
# }
# NOTE: LogSoftmax and ReLU do not expect any parameters
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
        
        h, w = conv_output_dims(input_height, input_width, layer["kernel_size"], layer["stride"])
        conv = nn.Conv2d(input_depth, layer["output_depth"], layer["kernel_size"], stride=layer["stride"])
        return [conv], kwargs(input_depth=layer["output_depth"], input_height=h, input_width=w, mode_cnn=True)
    
    elif layer_type == "AvgPool2d" or layer_type == "MaxPool2d":
        if not mode_cnn:
            raise ValueError
        ensure_not_none(input_height, input_width)
        ensure_not_in_dict(layer, "output_width", "output_depth")
        ensure_in_dict(layer, "kernel_size", "stride")
        
        h, w = pool_output_dims(input_height, input_width, layer["kernel_size"], layer["stride"])
        pool = (nn.AvgPool2d if layer_type == "AvgPool2d" else nn.MaxPool2d)(layer["kernel_size"], stride=layer["stride"], padding=0)
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
    # - NO padding (i.e. the image will shrink)
    # - NO dilation in convolutions or pools
    # - NO fancy pooling: only max pools and average pools with zero padding and symmetric strides
    # NOTE:
    # YOU are responsible for declaring ReLU and classification layers. YOU are responsible for knowing
    # what to stitch or not to stitch!
    def __init__(self, layers=None):
        super(Net, self).__init__()
        if layers is None:
            raise NotImplementedError
        
        # Tells us which layers (by indices in the array) are pools
        # so that when we stitch we can stitch across only with layers at the same pool
        # (if we so desire)
        self.pools_idxs = []

        # Input width will be re-used for fully connected layers to be the width of the linear layer
        kwargs = {
            "mode_cnn": True,
            "input_depth": 1,
            "input_height": 28,
            "input_width": 28,
        }

        # Layers are just nn layers... you must already know where to stitch!
        if not layers:
            raise NotImplementedError
        self.layers = []
        for layer in layers:
            to_append, kwargs = layers_from_layer_dict(layer, **kwargs)
            for l in to_append:
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

if __name__ == "__main__":
    shortnet = Net(layers=NET_3_2)
    longnet = Net(layers=NET_10_2)
    # TODO do something!
    # TODO make sure the sizes in examples.py are actually something desireable
    # TODO make stitching across these two guys!
    pass