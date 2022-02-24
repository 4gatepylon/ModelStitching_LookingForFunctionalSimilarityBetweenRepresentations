#
# This module implements a ResNet9 as described in the tutorial for FFCV:
# https://docs.ffcv.io/ffcv_examples/cifar10.html. You should be able to get up to
# at least 92% accuracy on CIFAR-10 with this model. We avoid using Autocast in this
# instance (even though they do it) because I have had NaN errors in the past using
# Autocast and Gradient Scaling for previous models (i.e. ResNet18).
#

from warnings import warn
from typing import List
from tqdm import tqdm
import numpy as np

import torch
import torchvision as tv
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveMaxPool2d, Linear, Module, Sequential
try:
    from ffcv.fields import IntField, RGBImageField
    from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.pipeline.operation import Operation
    from ffcv.transforms import RandomHorizontalFlip, Cutout, \
        RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
    from ffcv.transforms.common import Squeeze
    from ffcv.writer import DatasetWriter
except:
    warn("FFCV not installed. Your code may not run.")

BATCH_SIZE = 512
# Model is assumed to be on cuda:0
def trainResNet9(model, trainloader, testloader, epochs=24,):
    # Momentum SGD with a scheduler
    # The scheduler is used to adjust the learning rate over time, but I'm
    # not entirely sure what they are doing here.
    opt = SGD(model.parameters(), lr=.5, momentum=0.9, weight_decay=5e-4)
    iters_per_epoch = 50000 // BATCH_SIZE
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
        [0, 5 * iters_per_epoch, epochs * iters_per_epoch],
        [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)

    # Train in one shot since this should be pretty fast
    for ep in range(epochs):
        print(f"Epoch {ep}")
        for ims, labs in tqdm(trainloader):
            opt.zero_grad(set_to_none=True)
            out = model(ims)
            loss = loss_fn(out, labs)

            loss.backward()
            opt.step()
            scheduler.step()
    
    # Print out some evaluation (soon to be modularized away)
    model.eval()
    with torch.no_grad():
        total_correct, total_num = 0., 0.
        for ims, labs in tqdm(testloader):
            # Augment at test-time by taking the average of the prediction
            # (not the image) of the image and that of the image flipped left/right
            out = (model(ims) + model(torch.fliplr(ims))) / 2.
            total_correct += out.argmax(1).eq(labs).sum().cpu().item()
            total_num += ims.shape[0]
    print(f"Accuracy: {total_correct / total_num * 100:.1f}%")


class Mul(Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return Sequential(
        Conv2d(channels_in, channels_out, 
                        kernel_size=kernel_size, stride=stride, padding=padding,
                        groups=groups, bias=False),
        BatchNorm2d(channels_out),
        ReLU(inplace=True))

# NOTE: this is copied from ../long/cifar.py
def create_dataloaders(device):
  # NOTE this is from the FFCV tutorial
  
  # These statistics are with respect to the uint8 range, [0, 255]
  CIFAR_MEAN = [125.307, 122.961, 113.8575]
  CIFAR_STD = [51.5865, 50.847, 51.255]

  loaders = {}
  for name in ['train', 'test']:
    label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
    image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
    
    # Add image transforms and normalization 
    if name == 'train':
      image_pipeline.extend([
        RandomHorizontalFlip(),
        RandomTranslate(padding=2),
        Cutout(8, tuple(map(int, CIFAR_MEAN)))])
      
    image_pipeline.extend([
      ToTensor(),
      ToDevice(device, non_blocking=True),
      ToTorchImage(),
      # Note how we change it to a 32-bit integer to avoid casting and scaling which we suspect may
      # or may not have led to NaNs in previous training runs for other ResNets
      Convert(torch.float32),
      tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    loaders[name] = Loader(
        # NOTE that this is hardcoded due to the way we did it before in ../long/cifar.py
        f'../data/cifar_{name}.beton',batch_size=BATCH_SIZE,num_workers=8,
        order=OrderOption.RANDOM,
        drop_last=(name == 'train'),
        pipelines={'image': image_pipeline,'label': label_pipeline})
  return loaders

class ResNet9Cifar(Module):
    def __init__(self, num_classes=10):
        super(ResNet9Cifar, self).__init__()

        # I name them based on the output depth and width/height TODO
        self.conv64 = conv_bn(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv128 = conv_bn(64, 128, kernel_size=5, stride=2, padding=2)
        self.residual128 = Residual(Sequential(conv_bn(128, 128), conv_bn(128, 128)))
        self.conv256 = conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        self.maxpool256 = MaxPool2d(2),
        self.residual256 = Residual(Sequential(conv_bn(256, 256), conv_bn(256, 256)))
        self.conv128_later = conv_bn(256, 128, kernel_size=3, stride=1, padding=0)
        # Adaptive maxpool takes in the output size that we 
        # desire: https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool2d.html
        # In this case we want a 1x1 output (maximum of ALL along width/height) but we keep
        # a lot of depth.
        self.adamaxpool128_1x1 = AdaptiveMaxPool2d((1, 1))
        self.flattener = Flatten()
        self.linear128 = Linear(128, num_classes, bias=False)
        # I'm not sure why the FFCV tutorial scales everything down like this
        self.multiplier = Mul(0.2)
        self.runner = Sequential(
            self.conv64,
            self.conv128,
            self.residual128,
            self.conv256,
            self.maxpool256,
            self.residual256,
            self.conv128_later,
            self.adamaxpool128_1x1,
            self.flattener,
            self.linear128,
            self.multiplier)
    def forward(self, x):
        return self.runner(x)

if __name__ == "__main__":
    # This test is meant to run on cuda
    model = ResNet9Cifar()
    model = model.to(memory_format=torch.channels_last).cuda()
    trainloader, testloader = create_dataloaders(device='cuda:0')
    trainResNet9(model, trainloader, testloader, epochs=100)