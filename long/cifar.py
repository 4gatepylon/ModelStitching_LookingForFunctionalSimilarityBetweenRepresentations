# This is a CIFAR-10 FFCV sanity test runner. It trains and sets up basic stitches for FFCV.
# It follows the following tutorial: https://docs.ffcv.io/ffcv_examples/cifar10.html.
# Some parts are from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.

# Utility
from tqdm import tqdm
from argparse import ArgumentParser

# Pytorch for ML models and datasets
import torch
import torchvision as tv
import torch.optim as opt
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# The FFCV tutorial uses this to type-annotate
from typing import List

# FFCV is a library that enables very fast dataloading
# TODO add support for non-FFCV training (i.e. for local testing: do dummies)
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
  RandomHorizontalFlip,
  Cutout,
  RandomTranslate,
  Convert,
  ToDevice,
  ToTensor,
  ToTorchImage,
)
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

# Libraries to define and train/test networks (we re-write some of it for speed)
from net import Net
from examples_cnn import (
    NET_3_2 as C32,
    NET_4_2 as C42,
    NET_10_2 as C102,
)
# TODO Resnets, make sure to follow up with Yamini

from util import DATA_FOLDER
def download_datasets():
  # NOTE this is from the FFCV tutorial
  datasets = {
    'train': tv.datasets.CIFAR10(DATA_FOLDER, train=True, download=True),
    'test': tv.datasets.CIFAR10(DATA_FOLDER, train=False, download=True)
  }
  for (name, ds) in datasets.items():
    writer = DatasetWriter(f'{DATA_FOLDER}/cifar_{name}.beton', {'image': RGBImageField(),'label': IntField()})
    writer.from_indexed_dataset(ds)

# The batch size is shared in the train and test
BATCH_SIZE = 512
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
      Convert(torch.float16),
      tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    loaders[name] = Loader(
      f'{DATA_FOLDER}/cifar_{name}.beton',batch_size=BATCH_SIZE,num_workers=8,
      order=OrderOption.RANDOM,
      drop_last=(name == 'train'),
      pipelines={'image': image_pipeline,'label': label_pipeline})
  return loaders

# TODO first get training to work and the model to reach high accuracy and add the ability to save models
# Only AFTER that should you add stitching from the sanity we had going before... Maybe get resnets to work before.
# If you run out of time just enable MLP stitching since that in itself will also already be pretty interesting.
EPOCHS = 24
def run():
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  if use_cuda:
    print("Using Cuda (GPU)")
  else:
    print("Using CPU")

  # Get iterables that can go through the data with batches
  dataloaders = create_dataloaders(device)
  trainloader = dataloaders['train']
  testloader = dataloaders['test']

  # Classes is to help us with interpretability
  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  cifar_input_shape = (3, 32, 32)

  # We store it with format channels last according the FFCV tutorial
  model = Net(layers=C42, input_shape=cifar_input_shape)
  model = model.to(device, memory_format=torch.channels_last)

  # Loss function and optimizer
  criterion = F.cross_entropy
  optimizer = opt.SGD(model.parameters(), lr=0.001, momentum=0.9)
  scaler = GradScaler()

  # NOTE how we use fancy autocasting and grad scaling to enable differnt precisions
  # (that is from the FFCV tutorial)
  print(f"Training For {EPOCHS} epochs")

  # Train
  model.train()
  for epoch in range(1, EPOCHS+1):
    # Train one epoch
    for imgs, lbls in tqdm(trainloader):
      optimizer.zero_grad(set_to_none=True)
      with autocast():
        out = model(imgs)
        loss = criterion(out, lbls)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
  
  # Evaluate
  model.eval()
  with torch.no_grad():
    total_correct, total_num = 0., 0.
    for imgs, lbls in tqdm(testloader):
      with autocast():
        # Choose the average of the model of either the image or the image flipped
        out = (model(imgs) + model(torch.fliplr(imgs))) / 2.
        total_correct += out.argmax(1).eq(lbls).sum().cpu().item()
        total_num += imgs.shape[0]
    print(f'Accuracy: {total_correct / total_num * 100:.1f}%')

if __name__ == "__main__":
  parser = ArgumentParser(description='Decide what to run.')

  # Mode is 'download' or 'run'
  parser.add_argument('--mode', type=str)
  args = parser.parse_args()
  if args.mode == 'download':
    download_datasets()
  elif args.mode == 'run':
    run()
  else:
    raise ValueError
