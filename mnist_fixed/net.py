import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

class Net(nn.Module):
    # Note that MNIST has 28x28 images
    # convolutional depth is the number of convolutions (and they are put in the depth direction)
    # The width is the number of neurons of that layer (corresponds to the height of the weight
    # matrix that outputs that layer, and width of the matrix that takes that as an input)
    def __init__(self, conv1_depth=32, conv2_depth=64, hidden_width=128):
        super(Net, self).__init__()
        # MNIST is not rgb, but instead black and weight and so starts out with a depth of 1
        # We can mess with stride and/or convolutional window size later
        self.conv1 = nn.Conv2d(1, conv1_depth, 3, 1)
        self.conv2 = nn.Conv2d(conv1_depth, conv2_depth, 3, 1)

        # We can mess with these parameters later
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # We use 3x3 convolutions so the width and height lose 2 each convolution
        # meaning we go from 28 x 28 -> 26 x 26 -> 24 x 24. Subsequently, we half each
        # by using the max pool, yielding 12 x 12 (depth conv2_depth, still). Thus we end up with
        # 12 x 12 x conv2_depth = 144 * conv2_depth
        fc1_width = 144 * conv2_depth
        self.fc1 = nn.Linear(fc1_width, hidden_width)
        self.fc2 = nn.Linear(hidden_width, 10)

        # Store these to enable stitching later
        self.conv1_depth = conv1_depth
        self.conv2_depth = conv2_depth
        self.fc1_width = fc1_width

        # In the future we may want to use this
        self.hidden_width = hidden_width

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        latent = F.relu(x)
        return latent
    
    def classify(self, latent, use_dropout=False):
        if use_dropout:
            latent = self.dropout2(latent)
        x = self.fc2(latent)
        output = F.log_softmax(x, dim=1)
        return output

class StitchMode(Enum):
    CONV2 = 1 # Stitch before conv2
    FC1 = 2 # Stitch before fc1
    FC2 = 3 # Stitch before fc2

def linear_stitch_seq(starter_out, ender_in, device):
    return nn.Sequential(
        # Batch norm 1d subtracts the expected value (average), divides by the
        # standard deviation, and then rescales and biases that vector (once per mini-batch)
        nn.BatchNorm1d(starter_out),
        nn.Linear(starter_out, ender_in),
        nn.BatchNorm1d(ender_in),
    ).to(device)

def cnn_stitch_seq(starter_out, ender_in, device):
    # Note that connect_units is the output size of the previous layer (i.e. if we modify conv2
    # this would be the output size of conv1 and should yield the same output size, since conv2
    # is expecting that as input)
    return nn.Sequential(
        # Batch norm 2d does the same to batch-norm 1d but for a 2d tensor (i.e. a grid); I think you can
        # think of it as doing it on a flattened version, per mini-batch
        nn.BatchNorm2d(starter_out),
        # kernel size 1, stride 1
        nn.Conv2d(starter_out, ender_in, 1, 1),
        nn.BatchNorm2d(ender_in),
    ).to(device)

# Stitch net is very slow because of all the CPU logic, I think... 
# TODO we need a way to ascertain that starter and ender are, in fact,
# frozen
class StitchNet(nn.Module):
    def __init__(self, starter, ender, device, stitch_mode=None):
        super(StitchNet, self).__init__()
        if stitch_mode == StitchMode.CONV2:
            self.stitch_mode = StitchMode.CONV2
            # Second one will expect input to conv2 to have the output of its own convolution
            self.stitch = cnn_stitch_seq(starter.conv1_depth, ender.conv1_depth, device)
        elif stitch_mode == StitchMode.FC1:
            self.stitch_mode = StitchMode.FC1
            # Same as before second one will expect the input to have the dimensions of its own
            # inputs
            self.stitch = linear_stitch_seq(starter.fc1_width, ender.fc1_width, device)
        elif stitch_mode == StitchMode.FC2:
            self.stitch_mode = StitchMode.FC2
            # Once again we need to make sure to pass the inputs in the proper sizes that
            # the ender network expects
            self.stitch = linear_stitch_seq(starter.hidden_width, ender.hidden_width, device)
        else:
            raise NotImplementedError

        # Freeze the networks and store the frozen weights; careful with aliasing
        for param in starter.parameters():
            param.requires_grad = False
        for param in ender.parameters():
            param.requires_grad = False

        self.starter = starter
        self.ender = ender
    
    # Whether to forward and do the stitch right before conv2, right before fc1,
    # or whether to do it later for fc2 (or wherever), using only the starter for forward
    def forward_conv2(self, x):
        # Starter
        x = self.starter.conv1(x)
        x = F.relu(x)
        # Stitch
        x = self.stitch(x)
        # Ender
        x = self.ender.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.ender.dropout1(x)
        x = torch.flatten(x , 1)
        x = self.ender.fc1(x)
        latent = F.relu(x)
        return latent
    def forward_fc1(self, x):
        # Starter
        x = self.starter.conv1(x)
        x = F.relu(x)
        x = self.starter.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.starter.dropout1(x)
        x = torch.flatten(x, 1)
        # Stitch
        x = self.stitch(x)
        # Ender
        x = self.ender.fc1(x)
        latent = F.relu(x)
        return latent
    def forward(self, x):
        if self.stitch_mode == StitchMode.CONV2:
            return self.forward_conv2(x)
        elif self.stitch_mode == StitchMode.FC1:
            return self.forward_fc1(x)
        else:
            # Simply use the starter's forward to get the latent
            return self.starter(x)
    
    # Here we choose whether the starter or ender are giving us the latent for this classification
    def classify_starter(self, latent, use_dropout=False):
        if use_dropout:
            latent = self.starter.dropout2(latent)
        x = self.stitch(latent)
        x = self.ender.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    def classify(self, latent, use_dropout=False):
        if self.stitch_mode == StitchMode.FC2:
            return self.classify_starter(latent, use_dropout=use_dropout)
        return self.ender.classify(latent, use_dropout=use_dropout)