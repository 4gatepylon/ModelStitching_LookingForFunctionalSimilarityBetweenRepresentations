import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# This is for the overnight experiment
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from datetime import datetime

# These are two hard-coded examples with 3 and 10 convolutions respectively
# of sizes hard-coded by me. They both have 2 FCs followed by an FC classifier
# of the same shape/size
from examples_2 import NET_3_2, NET_10_2, NET_3_2_TO_NET_10_2_STITCHES

# Calculate the width and height of a convolutional layer
# from the input width and height as well as the (assumed symmetric)
# width and stride. Assume no padding. Same for pools.
def conv_output_dims(height, width, kernel, stride):
    # We take the floor since there is no padding
    output_height = math.floor((height - kernel + 1) / stride)
    output_width = math.floor((width - kernel + 1) / stride)
    return output_height, output_width
def pool_output_dims(height, width, kernel, stride):
    if stride != kernel:
        raise NotImplementedError("Tried to use pooling with stride {} != kernel {}".format(stride, kernel))
    # It's the same due to lack of padding
    return math.floor(height / stride), math.floor(width / stride)

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


# Given two layer indices find the stitch from the output of layer1 to the input
# if layer2. The output must be after a ReLU
# NOTE: gets the stitch to compare layer1 - 1 and layer2
def get_stitch(model1, model2, layer1_idx, layer2_idx):
    assert(type(model1) == Net)
    assert(type(model2) == Net)

    # We do not yet support stitching the first layer
    if layer1_idx == 0 or layer2_idx == 0:
        raise NotImplementedError

    # Find the last layer that provided dimensions
    # (so we know what dimensions we need to transform from)
    layer1 = model1.layers[layer1_idx]
    if type(layer1) in [nn.ReLU, nn.MaxPool2d, nn.AvgPool2d]:
        layer1 = model1.layers[layer1_idx - 1]
    # Sometimes there will be a pool after a ReLU which we support, but NOT two pools in a row
    if type(layer1) == nn.ReLU:
        layer1 = model1.layers[layer1_idx - 2]
    # And the last layer that provided dimensions for the recieving network
    # (i.e. we need to provide the same dimensions)
    layer2 = model2.layers[layer2_idx]
    if type(layer2) in [nn.ReLU, nn.MaxPool2d, nn.AvgPool2d]:
        layer2 = model2.layers[layer2_idx - 1]
    # Sometimes there will be a pool after a ReLU, which we support, but NOT
    # two pools in a row... it's assumed that this implies index >= 2
    if type(layer2) == nn.ReLU:
        layer2 = model2.layers[layer2_idx - 2]
    
    if type(layer1) == nn.Linear and type(layer2) == nn.Linear:
        # For linear, since we pass vectors as colums the format becomes (output dim, input dim)
        # and we want to input the output dim of layer1 and output the output dim of layer2
        input_dim, _ = layer1.weight.size()
        output_dim, _ = layer2.weight.size()
        return nn.Linear(input_dim, output_dim)
    
    elif type(layer1) == nn.Conv2d and type(layer2) == nn.Conv2d:
        # For convolution, the format is (output depth, input depth, kernel dim 1, kernel dim 2)
        # and we want to use kernel dims 1x1 and take the input depth as the output depth of
        # layer1 and the output depth as the output depth of layer2
        input_depth, _, _, _ = layer1.weight.size()
        output_depth, _, _, _ = layer2.weight.size()

        # Stride is 1 because we are selecting which filters we want
        # and note that nothing else can change the width or height since
        # either there was a ReLU after this (does nothing to the dimensions)
        # or we will raise NotImplementedError
        return nn.Conv2d(input_depth, output_depth, 1, stride=1)
        
    else:
        # We do not support linear -> conv, conv-> linear, or any other fancy things
        raise NotImplementedError("Tried to stitch (dims from) idx {} to idx {} of types {} and {}".format(
            layer1_idx, layer2_idx, type(layer1), type(layer2)))

# Given two models and a dictionary of indices
# {index in model 1: [list of indices in model2 that we want to try to stitch with]}
# return a dictionary {index in model 1 : {index in model 2: stitch layer}}
# NOTE: idx1 should always be the layer we output from and idx2 should be the layer we input to
def get_stitches(model1, model2, idxs):
    return {idx1 : {idx2 : get_stitch(model1, model2, idx1, idx2) for idx2 in idx2s} for idx1, idx2s in idxs.items()}

# Info per experiment
DEFAULT_TRAIN_BATCH_SIZE = 64
DEFAULT_TEST_BATCH_SIZE = 1000

DEFAULT_LR = 1.0
DEFAULT_LR_EXP_DROP = 0.7 # also called 'gamma'

# TODO scale up!
DEFAULT_EPOCHS_OG = 1
DEFAULT_EPOCHS_STITCH = 1

# Run 10 experiments
NUM_EXPERIMENTS = 40

# Simple train and test functions to run a single train or test run in a single epoch
def train(model, device, train_loader, optimizer):
    model.train()

    avg_loss = 0.0
    num_batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        avg_loss += torch.sum(loss).item()
        num_batches += 1
    
    num_batches = max(float(num_batches), 1.0)
    avg_loss /= num_batches
    return avg_loss
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    percent_correct = 100. * correct / len(test_loader.dataset)
    
    return test_loss, percent_correct

# Run train and test for each epoch that exists after initializing an optimizer and log to a file for
# further analysis in a future time
def train_og_model(model, model_name, device, train_loader, test_loader, epochs, logfile):        
    # Each model gets its own optimizer and scheduler since we may want to vary across them later
    optimizer = optim.Adadelta(model.parameters(), lr=DEFAULT_LR)
    scheduler = StepLR(optimizer, step_size=1, gamma=DEFAULT_LR_EXP_DROP)

    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer)
        test_loss, test_acc = test(model, device, test_loader)
        scheduler.step()

        # Log to the file
        with open(logfile, "a") as f:
            f.write("model {}\n\tepoch {}\n\ttrain loss {}\n\ttest loss{}\n\ttest acc{}\n".format(
                model_name, epoch, train_loss, test_loss, test_acc))
    # Make sure to save the model for further analysis
    torch.save(model.state_dict(), "{}.pt".format(model_name))

def train_stitch(model1, model2, stitch,
    model1_name, model2_name,
    idx1, idx2,
    device, train_loader, test_loader, epochs, logfile):
    stitched_model = StitchedForward(model1, model2, stitch, idx1, idx2)

    # NOTE how we only optimize the parameters in the stitch!
    for p in model1.parameters():
        p.requires_grad = False
    for p in model2.parameters():
        p.requires_grad = False

    optimizer = optim.Adadelta(stitch.parameters(), lr=DEFAULT_LR)
    scheduler = StepLR(optimizer, step_size=1, gamma=DEFAULT_LR_EXP_DROP)

    for epoch in range(1, epochs + 1):
        train_loss = train(stitched_model, device, train_loader, optimizer)
        test_loss, test_acc = test(stitched_model, device, test_loader)
        scheduler.step()
        # Log to the file
        with open(logfile, "a") as f:
            f.write("model {}\n\tepoch {}\n\ttrain loss {}\n\ttest loss{}\n\ttest acc{}\n".format(
                model_name, epoch, train_loss, test_loss, test_acc))
    # Make sure to save the model for further analysis
    torch.save(model.state_dict(), "stitch_{}_l{}_to_{}_l{}.pt".format(model1_name, idx1, model2_name, idx2))

if __name__ == "__main__":
    # Make a folder for these experiments
    date_prefix = datetime.now().strftime("%Y-%m-%d")
    if not os.path.isdir("experiments"):
        print("Making experiments folder")
        os.mkdir("experiments")
    if os.path.isdir("experiments/{}".format(date_prefix)):
        raise RuntimeError("An experiment is already running for {}".format(date_prefix))
    print("Making experiment for date {}".format(date_prefix))
    os.mkdir("experiments/{}".format(date_prefix))

    # Initialize the datasets (assuming already downloaded)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device is {}".format("cuda" if use_cuda else "cpu"))

    train_kwargs = {'batch_size': DEFAULT_TRAIN_BATCH_SIZE}
    test_kwargs = {'batch_size': DEFAULT_TEST_BATCH_SIZE}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    train_kwargs['shuffle'] = True
    test_kwargs['shuffle'] = True
    
    # load the dataset for test and train from a local location (supercloud has to access to the internet)
    dataset1 = datasets.MNIST('./data', train=True, download=False, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Run various experiments, each storing an output in a folder that is aptly named for your understanding
    for experiment_num in range(NUM_EXPERIMENTS):
        exp_name = "experiments/{}/{}".format(date_prefix, experiment_num)
        os.mkdir(exp_name)
        print("Storing experiment {} in {}".format(experiment_num, exp_name))
        print("*** Running experiment ***")
        
        # NOTE: all these will be stored in .pt files in the two training runs
        print("*** Initializing nets ***")
        shortnet = Net(layers=NET_3_2).to(device)
        longnet = Net(layers=NET_10_2).to(device)
        print("*** Initializing stitches ***")
        stitches = get_stitches(shortnet, longnet, NET_3_2_TO_NET_10_2_STITCHES)

        # NOTE: this will log og accuracy!
        print("*** Training og nets ***")
        train_og_model(shortnet, "shortnet", device, train_loader, test_loader, DEFAULT_EPOCHS_OG, "shortnet_train.txt")
        train_og_model(longnet, "longnet", device, train_loader, test_loader, DEFAULT_EPOCHS_OG, "longnet_train.txt")

        # NOTE: this will log stitching accuracy!
        print("*** Training stitches ***")
        for idx1, idx2s in stitches.items():
            for idx2, stitch in idx2s.items():
                train_stitch(
                    shortnet, longnet, stitch,
                    "shortnet", "longnet",
                    idx1, idx2, device,
                    train_loader, test_loader, DEFAULT_EPOCHS_STITCH, "shortnet_l{}_longnet_l{}.txt".format(idx1, idx2))

    ### Important after we are done with overnight training
    # TODO find a way to analyze based on the accuracies
    # TODO find a way to avoid padding and stitch things with different shapes
    pass