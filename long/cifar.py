# This is where we are gong to do sanity testing with CIFAR
# Specifically, that means that we are going to use the CIFAR dataset because we think that
# MNIST is not really sufficient to capture relevant differences between layers (or that
# the transforms that we use for stitches, even when simple like 1x1 convolutions, are too
# complicated for the dataset)

# Importantly, we intend to use FFCV (it's a fast dataloader library that
# should vastly lower our training time... this is important to get the report to Yamini soon)
# and some form of CNN or ResNet (i.e. ResNet18).
# We will have to check with the shapes of the Resnet
# since it may be desireable to change the shape to
# enable simpler stitching. Alternatively,
# we may choose to use some sort of different stitch,
# though that runs contrary to what we'd like.

# TODO make sure our CNNs have the right shape (that may already be possible not sure)
# TODO make sure that our loss function is OK
# TODO ensure that we are stitching correctly, make sure that asserts are one-liners (and 
#      call a function if they must, since this will enable -O assertion removal without keeping
#      unused looping)

import torch
import torch.nn as nn
from typing import List

import torch as ch
import torchvision as tv

# TODO get rid of this
from torchvision import transforms

# TODO speed
# this may only work on supercloud in the env...
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

from net import (
    Net, 
)
from training import (
  train1epoch,
  test
)

import torch.optim as optim

# TODO actually use init_cifar
# TODO stitching and other shapes
from examples_cnn import (
    NET_3_2 as C32,
)

# TODO resnets
# TODO hyperparameters

def main():
  print("transforms being defined")
  transform = transforms.Compose(
    [transforms.ToTensor(),
    # normalize to mean 0.5 and standard deviation 0.5 along all dimensions
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  batch_size = 4

  print("creating datasets")
  # TODO make this the init_cifar method in training.py
  trainset = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
  testset = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

  print("network etc")
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  cifar_input_shape = (3, 32, 32)

  net = Net(layers=C32, input_shape=cifar_input_shape)
  criterion = nn.CrossEntropyLoss()
  opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  # try training for 10 epochs
  print("train 10 epochs")
  for _ in range(10):
    train1epoch(net, device, trainloader, opt, criterion=criterion)
    _, acc = test(net, device, testloader, criterion=criterion)
    print("acc is {}".format(acc))
  print("done")

if __name__ == "__main__":
  main()
