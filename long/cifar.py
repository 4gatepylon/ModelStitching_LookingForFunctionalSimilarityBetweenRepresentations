# This is a CIFAR-10 FFCV sanity test runner. It trains and sets up basic stitches for FFCV.
# It follows the following tutorial: https://docs.ffcv.io/ffcv_examples/cifar10.html.
# Some parts are from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.

# Utility
from warnings import warn
from tqdm import tqdm
from argparse import ArgumentParser
from datetime import datetime

# Used for plotting and calculations with our results
import matplotlib.pyplot as plt
import numpy as np

# Torchinfo is great for visualizing shapes!
# It doesn't work for me when I run python, but it does work
# in the interpreter... wierd! I've pasted a resnet
# summary at the very end. Try this (batch size 1 for speed):
#
# from torchinfo import summary
# model = resnet18k_cifar(k=64, num_classes=10)
# summary(model, (1, 3, 32, 32))

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
try:
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
except:
  warn("Failed to find FFCV, might not be able to run")

# Libraries to define and train/test networks (we re-write some of it for speed)
from net import Net
from examples_cnn import (
    NET_3_2 as C32,
    NET_4_2 as C42,
    NET_10_2 as C102,
)
# Yamini's ResNet18 for CIFAR (we are trying to get a model that can get high acc)
from resnet import resnet18k_cifar

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
  use_resnet = True
  model = resnet18k_cifar(num_classes=len(classes)) if use_resnet else Net(layers=C102, input_shape=cifar_input_shape)
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


# This will generate and save a matrix heatmap to a png file
# You will want to tell it what rows and columns correspond to, since
# often the layers will be skipped. The matrix is expected to be square.
# All values should be in [0, 1] and for layers that could not be stitched
# you are encouraged to put zero.
def matrix_heatmap(mat=None, Xs=None, Ys=None, decimals=2, model_name="random_mat", output_file_name=None, flipy=True, show=False):

  # Check that the matrix is valid, and generate a random one if none is provided
  # (this is functionality for debugging the matrix_heatmap) function in itself
  mat_height, mat_width = None, None
  if mat is None:
    warn("Trying to matrix heatmap no matrix, will generate random matrix with high diagonal")

    mat_height, mat_width = 7, 7
    mat = np.random.uniform(0, 0.5, size=(mat_height, mat_width))

    assert mat_width == mat_height
    for i in range(mat_height):
      mat[i, i] += 0.5
  else:
    assert type(mat) == torch.Tensor or type(mat) == np.ndarray
    if type(mat) == torch.Tensor:
      mat = mat.numpy()

    # Has to be a square 2D matrix
    assert len(mat.shape) == 2
    assert mat.shape[0] == mat.shape[1]

    mat_height, mat_width = mat.shape
  
  # Round for readability
  if not decimals is None:
    mat = np.round(mat, decimals=decimals)
  
  # These ticks don't seem to do much, but they are in the example I
  # copied from...
  yticks, xticks = np.arange(mat_height), np.arange(mat_width)
  if Ys is None:
    Ys = yticks
  if Xs is None:
    Xs = xticks
  
  # Often we will want to flip the matrix vertically so as to have
  # the origin be in the bottom left instead of the top left. By default
  # we will do this, but some may prefer to disable it.
  if flipy:
    Ys = np.flip(Ys)
    mat = np.flip(mat, axis=0)

  # Copied from
  # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
  # (go down to #sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py)
  fig, ax = plt.subplots()
  im = ax.imshow(mat)

  ax.set_yticks(yticks)
  ax.set_xticks(xticks)
  ax.set_xticklabels(Xs)
  ax.set_yticklabels(Ys)
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

  # This is inserting the text into the boxes so that we can compare easily
  for i in range(len(Xs)):
      for j in range(len(Ys)):
          text = ax.text(j, i, mat[i, j], ha="center", va="center", color="w")

  ax.set_title("Self Similarity of {}".format(model_name))
  fig.tight_layout()
  if show:
    plt.show()
  else:
    # We will save the image grid as
    # self_sim_resnet18_2022_02_16_9_45 for example
    if output_file_name is None:
      output_file_name = "self_sim_{}_{}.png".format(model_name, datetime.now().strftime("%Y_%m_%d_%H_%M"))
    plt.savefig(output_file_name)

  # This clears matplotlib to keep plotting if we want
  plt.clf()

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
