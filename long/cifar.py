# This is a CIFAR-10 FFCV sanity test runner. It trains and sets up basic stitches for FFCV.
# It follows the following tutorial: https://docs.ffcv.io/ffcv_examples/cifar10.html.
# Some parts are from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.

# Utility
import os
import json
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

from tensorboardX import SummaryWriter

# The FFCV tutorial uses this to type-annotate
from typing import List

# FFCV is a library that enables very fast dataloading
# NOTE we will want to add support for non-FFCV training (i.e. for local testing: do dummies)
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

# We have a 3-layer conv followed by a 5-layer MLP that we will test
from net import (
  Net,
  get_stitches as get_stitches_raw,
  StitchedForward,   
)
from c35 import (
  NET_3_5 as c35_net,
  NET_3_5_TO_NET_3_5_STITCHES_INTO_FMT as c35_stitches_into_fmt,
)
# As well as Yamini's ResNet18 for CIFAR (we are trying to get a model that can get high acc)
from resnet import (
  resnet18k_cifar,
  RESNET18_SELF_STITCHES_COMPARE_FMT as resnet_stitches_compare_fmt,
  stitches_resnet
)
from training import train1epoch
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

def eval_model(model, testloader):
  model.eval()
  with torch.no_grad():
    total_correct, total_num = 0., 0.
    for imgs, lbls in tqdm(testloader):
      with autocast():
        # Choose the average of the model of either the image or the image flipped
        out = (model(imgs) + model(torch.fliplr(imgs))) / 2.
        total_correct += out.argmax(1).eq(lbls).sum().cpu().item()
        total_num += imgs.shape[0]
  return total_correct / total_num

STITCH_TRAIN_EPOCHS = 100
TRAIN_EPOCHS = 200
PRINT_EVERY = 20

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
CIFAR_INPUT_SHAPE = (3, 32, 32)
CIFAR_FOLDER = "cifar_out"

def scale_train_model(model, optimizer, criterion, scaler, trainloader, testloader, epochs=TRAIN_EPOCHS, print_every=PRINT_EVERY, tensorboardx_writer=None, tensorboardx_scalar=None):
  model.train()
  for epoch in range(0, epochs):
    if epoch % print_every == 0:
      acc = eval_model(model, testloader)
      # NOTE we'll want to add loss, etcetera...
      if tensorboardx_scalar and tensorboardx_writer:
        tensorboardx_writer.add_scalar(tensorboardx_scalar, acc)
      print(f"Epoch {epoch}, Accuracy: {acc * 100:.1f}%")
      model.train()
    # NOTE there is a train1epoch function but it does not use gradient scaling
    # which is probably not necessary, but the tutorial used it and in theory it
    # should help with numerical errors...
    for imgs, lbls in tqdm(trainloader):
      optimizer.zero_grad(set_to_none=True)
      with autocast():
        out = model(imgs)
        loss = criterion(out, lbls)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

TENSORBOARDX_TRAIN_FOLDER = "tensorboard_train"
TENSORBOARDX_STITCH_TRAIN_FOLDER = "tensorboard_stitch_train"
def train_networks(model_names=["ResNet18", "C35"]):
  assert len(model_names) == 2
  assert model_names[0] == "ResNet18" and model_names[1] == "C35"

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

  # We store it with format channels last according the FFCV tutorial
  cnn = resnet18k_cifar(num_classes=len(CLASSES))
  mlp = Net(layers=c35_net, input_shape=CIFAR_INPUT_SHAPE)
  cnn = cnn.to(device, memory_format=torch.channels_last)
  mlp = mlp.to(device, memory_format=torch.channels_last)

  # Loss function and optimizer
  criterion = F.cross_entropy
  optimizer = opt.Adam(cnn.parameters(), lr=0.01, momentum=0.9)
  scaler = GradScaler()

  # NOTE how we use fancy autocasting and grad scaling to enable differnt precisions
  # (that is from the FFCV tutorial)
  # NOTE you will want to pipe each of these into some text file
  for model, model_name in [(cnn, model_names[0]), (mlp, model_names[0])]:
    print(f"Training {model_name} For {TRAIN_EPOCHS} epochs")
    
    writer = SummaryWriter(TENSORBOARDX_TRAIN_FOLDER)
    tname = model_name
    scale_train_model(model, optimizer, criterion, scaler, trainloader, testloader, epochs=TRAIN_EPOCHS, print_every=PRINT_EVERY, tensorboardx_writer=writer, tensorboardx_scalar=tname)
    
    print(f"Final Accuracy: {eval_model(model, testloader) * 100:.1f}%") 

    fname = os.path.join(CIFAR_FOLDER, f"{model_name}.pt")
    print(f"Saving to {fname}")
    torch.save(model.state_dict(), fname)
  
def stitches_mlp(model1, model2, validst, device):
  # This is a nested map
  stitches = get_stitches_raw(model1, model2, validst, device)
  # These are basically indices
  keyset = set()
  for l1, l2s in stitches.items():
    for l2 in l2s.keys():
      keyset.add(l1)
      keyset.add(l2)
  
  # Now we have a mapping of index to layer
  idx2layer = sorted(list(keyset))
  layer2idx = {idx2layer[i] : i for i in range(len(idx2layer))}

  # Here we put None everywhere that there wasn't a stitch pairing and elsewhere
  # we put the stitch. This gives us the same format as the resnet, which makes
  # it easier later to do table visualization since we can do maps (i.e. a table
  # map is just a map of maps and the inner map is just -> 0 if None else acc of the guy)
  fw = lambda l1, l2: StitchedForward(model1, model2, stitches[l1][l2], l1, l2)
  of = lambda l1, l2: fw(l1, l2) if l1 in stitches and l2 in stitches[l1] else None
  table = [[of(l1, l2) for l1 in range(len(idx2layer))] for l2 in range(len(idx2layer))]
  return table, idx2layer, layer2idx

# NOTE this function is basically 1 epoch of training
# where all it does it confirm that gradients update properly
def sanity_test_stitched_model(stitched_model, optimizer, device, train_loader, same_model=True):
  sane = True
  fail = "Sanity Test Fail"

  # Check that the number of parameters in the stitched network is correct (model1 + model2 + stitch
  # if they are not the same model and otherwise model1 + stitch). Also check that they are non-zero)
  stitch = stitched_model.get_stitch()
  model1 = stitched_model.get_model1()
  model2 = stitched_model.get_model2()

  num_stitched_model_params = len(stitched_model.parameters())
  num_stitch_params = len(stitch.parameters())
  num_model1_params = len(model1.parameters())
  num_model1_params = len(model2.parameters())

  if num_stitched_model_params <= 0:
    warn(f"{fail}: Stitched model has {num_stitched_model_params} parameters (should be more than zero)")
    sane = False
  if num_stitch_params <= 0:
    warn(f"{fail}: Stitch has {num_stitch_params} parameters (should be more than zero)")
    sane = False
  if num_model1_params <= 0:
    warn(f"{fail}: Model 1 has {num_model1_params} parameters (should be more than zero)")
    sane = False
  if num_model1_params <= 0:
    warn(f"{fail}: Model 2 has {num_model1_params} parameters (should be more than zero)")
    sane = False
  
  total_params = num_model1_params + num_stitch_params
  if same_model and total_params != num_stitched_model_params:
    warn(f"{fail}: Self-stitch, but number of stitched model parameters is {num_stitched_model_params}, when it should be {total_params} = (model 1) {num_model1_params} + (stitch) {num_stitch_params}")
    sane = False
  total_params += num_model1_params
  if not same_model and total_params != num_stitched_model_params:
    warn(f"{fail}: Non-self-stitch, but number of stitched model parameters is {num_stitched_model_params}, when it should be {total_params} = (model1) {num_model1_params} + (stitch) {num_stitch_params} + (model2) {num_model1_params}")
    sane = False

  if not sane:
    return False
  
  # Check that the after training one epoch
  #  -> the weights of the original networks have not changed
  #  -> the weights of the stitch have changed
  # (using a generic optimizer)

  # Helper methods to copy a model's parameters and check that two lists are the same
  # NOTE that we'll want to modularize this out pretty soon if we want this to stop being hell
  pclone = lambda model: [p.data.detach().clone() for p in model.parameters()]
  listeq = lambda l1, l2: min((torch.eq(a, b).int().min().item() for a, b in zip(l1, l2))) == 1
  
  model1_params, model2_params, stitch_params = pclone(model1), pclone(model2), pclone(stitch)

  # We do not care about gradient scaling since we just want to see that the right things change
  # and/or do NOT change. We do not care so much that we optimize the weights properly
  train1epoch(stitched_model, device, train_loader, optimizer, criterion=F.cross_entropy)

  model1_params_same, model2_params_same, stitch_params_changed = pclone(model1), pclone(model2), pclone(stitch)
  if not listeq(model1_params, model1_params_same):
    warn(f"{fail}: Model 1 was updated by stitch training")
    sane = False
  if not listeq(model2_params, model2_params_same):
    warn(f"{fail}: Model 2 was updated by stitch training")
    sane = False
  if listeq(stitch_params, stitch_params_changed):
    warn(f"{fail}: Model 3 was updated by stitch training")
    sane = False
  
  return sane

# Check that every stitch INSIDE each stitched_model
# has a different pointer for every paramemter. We TRUST
# that EVERY MODEL HAS THE CORRECT POINTERS (that is to say
# the same model will have the same pointers, but different
# for the differnet layers IFF they are supposed to and different
# models will have differnet pointers IFF they are supposed to)
def sanity_test_stitches_ptrs(stitched_models):
  layers = range(len(stitched_models))

  sane = True
  for l1a in layers:
    for l2a in layers:
      model_a = stitched_models[l1a][l2a]
      if not model_a is None:
        for l1b in layers:
          for l2b in layers:
            model_b = stitched_models[l1b][l2b]
            if not model_b is None:
              model_a_st = model_a.get_stitch()
              model_b_st = model_b.get_stitch()
              same_stitch = l1a == l1b and l2a == l2b
              # We want check to be true if NOT same stitch
              # and check to be false otherwise, so we XOR with
              # same stitch to flip it only if same_stitch is True
              equals = lambda check: same_stitch ^ check
              for p_a_idx, p_a in enumerate(model_a_st.parameters()):
                  for p_b_idx, p_b in enumerate(model_b_st.parameters()):
                    if not equals(p_a.data.data_ptr(), p_b.data.data_ptr()):
                      err = "not equal" if same_stitch else "equal"
                      warn(f"Sanity Ptr Test Fail: Stitches from layers({l1a}->{l2a})param({p_a_idx}) and layers({l1b}-{l2b})param({p_b_idx}): ptrs are {err} but should not be")
                      sane = False
  return sane

# NOTE we may not always which to clip the sims
def train_stitch(stitched_model, device, trainloader, testloader, same_model=True, model1_orig_acc=None, model2_orig_acc=None, min_sim=0.0, max_sim=1.0):
  # NOTE that perhaps this is a bit agressive an LR but hey... idt it matters...
  criterion = F.cross_entropy
  optimizer = opt.Adam(stitched_model.get_stitch().parameters(), lr=0.01, momentum=0.9)
  scaler = GradScaler()

  # NOTE that due to the code structure this will run on every stitch which is OK for testing
  assert sanity_test_stitched_model(stitched_model, optimizer, device, trainloader, same_model=same_model)

  writer = SummaryWriter(TENSORBOARDX_STITCH_TRAIN_FOLDER)
  tname = f"{stitched_model.get_layer1()}->{stitched_model.get_layer2()}"
  scale_train_model(stitched_model, optimizer, criterion, scaler, trainloader, testloader, epochs=STITCH_TRAIN_EPOCHS, print_every=PRINT_EVERY, tensorboardx_writer=writer, tensorboardx_scalar=tname)

  acc = eval_model(stitched_model, testloader)
  print(f"Final Stitch Accuracy: {acc * 100:.1f}%")

  sim = acc

  can_penalty = not (model1_orig_acc is None or model2_orig_acc is None)
  if can_penalty:
    err = 1 - acc
    # Minimum error of a previous model to avoid bottlenecks
    # NOTE we may want to the option to add averaging and maxing mode
    min_err_bias = 1 - max(model1_orig_acc, model2_orig_acc)
    penalty = err - min_err_bias
    sim = 1 - penalty
  sim = min(max(sim, min_sim), max_sim)

  text = "" if can_penalty else "not"
  print(f"Final sim ({text} using penalty) is {sim * 100:.1f}")

  return sim

MLP_IDX2LAYER_FNAME = "C35_idx2layer.json"
def train_stitches(model_names=["ResNet18", "C35"]):
  assert len(model_names) == 2
  assert model_names[0] == "ResNet18" and model_names[1] == "C35"

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

  # Initialize new models and load from previously saved models (this does not expect to run at the
  # exact same time as training)
  cnn = resnet18k_cifar(num_classes=len(CLASSES))
  mlp = Net(layers=c35_net, input_shape=CIFAR_INPUT_SHAPE)
  fname0 = os.path.join(CIFAR_FOLDER,f"{model_names[0]}.pt")
  fname1 = os.path.join(CIFAR_FOLDER, f"{model_names[1]}.pt")
  cnn.load_state_dict(torch.load(fname0))
  mlp.load_state_dict(torch.load(fname1))

  # Sanity check that these are loaded properly (should have high accuracy)
  cnn_acc = eval_model(cnn, testloader)
  mlp_acc = eval_model(mlp, testloader)
  print(f"ResNet18 Accuracy: {cnn_acc * 100:.1f}%")
  print(f"C35 Accuracy: {mlp_acc * 100:.1f}%")
  assert cnn_acc > 0.8
  assert mlp_acc > 0.4

  # Put into the proper device
  cnn = cnn.to(device, memory_format=torch.channels_last)
  mlp = mlp.to(device, memory_format=torch.channels_last)
  # Load random ones for comparison. 
  cnn_rand = resnet18k_cifar(num_classes=len(CLASSES)).to(device, memory_format=torch.channels_last)
  mlp_rand = Net(layers=c35_net, input_shape=CIFAR_INPUT_SHAPE).to(device, memory_format=torch.channels_last)

  # rstitches is a StitchedResnet[5][5]
  rstitches_from = lambda starter: stitches_resnet(
    starter, cnn, 
    valid_stitches=resnet_stitches_compare_fmt,
    in_mode="outfrom_into", out_mode="compare", 
    min_layer=0, max_layer=5,
    device=device)

  # NOTE we pick to use the random network as prefix, but either should work
  rstitches = rstitches_from(cnn)
  rstitches_rand = rstitches_from(cnn_rand)

  # Because the MLP layers are not nominally 1:1 with the indices, we also store the layer (which in
  # c35.py is explained: i.e. which layer corresponds to what portion of the computation graph)
  mstitches, midx2layer, _ = stitches_mlp(mlp, mlp, c35_stitches_into_fmt, device)
  mstitches_rand, _, _ = stitches_mlp(mlp_rand, mlp, c35_stitches_into_fmt, device)

  # Make sure these are square tables/matrices
  assert len(mstitches) > 0
  assert len(rstitches) > 0
  assert len(mstitches) == max(map(lambda st: len(st), mstitches))
  assert len(mstitches) == min(map(lambda st: len(st), mstitches))
  assert len(rstitches) == max(map(lambda st: len(st), rstitches))
  assert len(rstitches) == max(map(lambda st: len(st), rstitches))

  # Make sure that they have the proper pointers (i.e. stitches are different)
  assert sanity_test_stitches_ptrs(rstitches)
  assert sanity_test_stitches_ptrs(rstitches_rand)
  assert sanity_test_stitches_ptrs(mstitches)
  assert sanity_test_stitches_ptrs(mstitches_rand)

  mN = len(mstitches)
  rN = len(rstitches)

  # These are the matrices of similiarities. We calculate them by
  # basically 
  mstitch_sims = [[0.0 for _ in range(mN)] for _ in range(mN)]
  rstitch_sims = [[0.0 for _ in range(rN)] for _ in range(rN)]
  mstitch_rand_sims = [[0.0 for _ in range(mN)] for _ in range(mN)]
  rstitch_rand_sims = [[0.0 for _ in range(rN)] for _ in range(rN)]

  # NOTE in the future we will prefer to have mstitch_sims in mode compare for interpretability
  for l1 in range(mN):
    for l2 in range(mN):
      if not mstitches[l1][l2] is None:
        # NOTE that the train function has a side effect: the stitch weights are updated
        mstitch_sims[l1][l2] = train_stitch(mstitches[l1][l2], device, trainloader, testloader, same_model=True)
        mstitch_rand_sims[l1][l2] = train_stitch(mstitches_rand[l1][l2], trainloader, testloader, same_model=False)
        # NOTE we do not store the radom "control" since we just want it to debias our
        # accuracies
        fname = os.path.join(CIFAR_FOLDER, f"mlp_stitch_{l1}_{l2}_mode_compare.pt")
        fname_mod = os.path.join(CIFAR_FOLDER, f"mlp_stitched_model_{l1}_{l2}_mode_compare.pt")
        torch.save(rstitches[l1][l2], fname_mod)
        torch.save(rstitches[l1][l2].stitch, fname)
  
  for l1 in range(rN):
    for l2 in range(rN):
      if not rstitches[l1][l2] is None:
        # NOTE that the train function has a side-effect: the stitch weights are updated
        rstitch_sims[l1][l2] = train_stitch(rstitches[l1][l2], device, trainloader, testloader, same_model=True)
        rstitch_rand_sims[l1][l2] = train_stitch(rstitches_rand[l1][l2], trainloader, testloader, same_model=False)
        # NOTE we do not store the radom "control" since we just want it to debias our
        # accuracies
        fname = os.path.join(CIFAR_FOLDER, f"resnet_stitch_{l1}_{l2}_mode_compare.pt")
        fname_mod = os.path.join(CIFAR_FOLDER, f"resnet_sitched_model_{l1}_{l2}_mode_compare.pt")
        torch.save(rstitches[l1][l2], fname_mod)
        torch.save(rstitches[l1][l2].stitch, fname)

  debias = lambda sim, rand_sim: [[sim[l1][l2]-rand_sim[l1][l2] for l2 in range(len(sim))] for l1 in range(len(sim))]
  rstitch_debias_sims = debias(rstitch_sims, mstitch_rand_sims)
  mstitch_debias_sims = debias(mstitch_sims, mstitch_rand_sims)
  sims = {
    model_names[0] : {
      "sim": rstitch_sims,
      "rand_sim": rstitch_rand_sims,
      "debias_sim": rstitch_debias_sims,
    },
    model_names[1]: {
      "sim": mstitch_sims,
      "rand_sim": mstitch_rand_sims,
      "debias_sim": mstitch_debias_sims,
    }
  }
  for model_name, suffix2sim in sims.items():
    for suffix, sim in suffix2sim.items():
      fname = "_".join([model_name, suffix, "tensor.pt"])
      path = os.path.join(CIFAR_FOLDER, fname)
      torch.save(torch.tensor(sim), path)
      
      rounded = np.round(np.array(mstitch_sims), decimals=2)
      print(f"*** {fname} ***")
      print(rounded)
  
  fmidx2layer = os.path.join(CIFAR_FOLDER, MLP_IDX2LAYER_FNAME)
  with open(fmidx2layer, 'w') as fp:
    json.dump(midx2layer, fp)

# This will generate and save a matrix heatmap to a png file
# You will want to tell it what rows and columns correspond to, since
# often the layers will be skipped. The matrix is expected to be square.
# All values should be in [0, 1] and for layers that could not be stitched
# you are encouraged to put zero.
def matrix_heatmap(mat=None, Xs=None, Ys=None, decimals=2, model_name="random_mat", input_file_name=None, output_file_name=None, flipy=True, show=False):

  # Check that the matrix is valid, and generate a random one if none is provided
  # (this is functionality for debugging the matrix_heatmap) function in itself
  mat_height, mat_width = None, None
  if mat is None and input_file_name is None:
    warn("Trying to matrix heatmap no matrix, will generate random matrix with high diagonal")

    mat_height, mat_width = 7, 7
    mat = np.random.uniform(0, 0.5, size=(mat_height, mat_width))

    assert mat_width == mat_height
    for i in range(mat_height):
      mat[i, i] += 0.5
  else:
    mat = torch.load(input_file_name)

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


def create_heatmaps(model_names=["ResNet18", "C35"]):
  assert len(model_names) == 2
  assert model_names[0] == "ResNet18" and model_names[1] == "C35"

  suffixes = ["sim", "rand_sim", "debias_sim"]

  # We'll use this to label the Xs and Ys
  midx2layer = None
  with open(MLP_IDX2LAYER_FNAME, "r") as f:
    midx2layer = json.load(f)
  if midx2layer is None:
    warn(f"Might fail: could not load mlp idx2layer file {MLP_IDX2LAYER_FNAME}")
  
  for model_name in model_names:
    for suffix in suffixes:
      fname = "_".join([model_name, suffix, "tensor"])
      fname_in = fname + ".pt"
      fname_out = fname + ".png"
      Xs = None
      Ys = None
      if model_name == "C35":
        # Rember that this dictionary maps indices to layers
        Xs = list(midx2layer.values())
        Ys = list(midx2layer.values())
      matrix_heatmap(
        mat=None, Xs=Xs, Ys=Ys, decimals=2,
        model_name=f"{model_name} {suffix}",
        input_file_name=fname_in, output_file_name=fname_out,
        flipy=False, show=False)

if __name__ == "__main__":
  # matrix_heatmap(Xs=["a", "b", "c", "d", "e", "f", "g"])
  parser = ArgumentParser(description='Decide what to run.')
  # Mode is 'download' or 'run'
  parser.add_argument('--mode', type=str)
  args = parser.parse_args()
  if args.mode == 'download':
    download_datasets()
  elif args.mode == 'train_models':
    train_networks(model_names=["ResNet18", "C35"])
  elif args.mode == 'train_stitches':
    train_stitches(model_names=["ResNet18", "C35"])
  elif args.mode == 'create_heatmaps':
    create_heatmaps(model_names=["ResNet18", "C35"])
  else:
    raise ValueError
