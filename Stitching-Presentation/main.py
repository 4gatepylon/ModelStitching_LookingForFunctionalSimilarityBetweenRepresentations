import torch.nn as nn
import torch.nn.functional as F

# Simplest possible scenario from the paper: simply train on
# CIFAR-10 dataset using a ResNet18 with the parameters they give.
class CifarResetNet18(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

# Train on CIFAR-10 Dataset
def train_cifar10(model):
    raise NotImplementedError

# Stitch with a 1x1 convolutional layer (i.e. just rescale all the inputs)
def stitch(model_left, model_right, stitch_layer_idx):
    raise NotImplementedError

# Freeze the non-stitch weights and train on the joint stitched model with CIFAR-10
def train_stitch_cifar10(stitched_model):
    raise NotImplementedError

# Train and cache (or load from cache) a couple ResNet-18 models for CIFAR-10. Seed differently.
# Then stitch them on a range of layer indices and train the resulting network with the non-stitch-layer
# weights frozen. Evaluate the output and create a simple table of results.
def main():
    raise NotImplementedError

if __main__ == "__main__":
    main()