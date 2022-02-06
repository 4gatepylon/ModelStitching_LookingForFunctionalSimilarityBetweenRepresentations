# What is this?
I wrote a short script here to train a small stitching model for the AIM presentation.

I will put this on a gist and it should be available for everyone seeing this.

This does not do anything other than the simplest models from the paper.

# What does this do?
## Stitching Modes
- CNN: simple 1x1 convolutions
- FC: simply a linear layer

## Datasets
- CIFAR-10 with a ResNet-18

```
adapted to CIFAR-10 dimensions with 64 filters in the first convolutional layer.
We also train a wider ResNet-w2x and narrower ResNet-0.5x with 128 and 32 filters in the first layer respectively.
For the deep ResNet, we train a ResNet-164.
```

```
For experiments with changing label distribution, we also train the base ResNet-18. As specified, for
the binary Object-Animal labels, we group Cat, Dog, Frog, Horse, Deer as label 0 and Truck,
Ship, Airplane, Automobile as label 1. For the label noise experiments, with probability
p = {0.1,0.5,1.0}we assign a random label to p fraction of the training set.
```

```
All the ResNets are trained for 64K gradient steps of SGD with 0.9 momentum, a starting learn-
ing rate of 0.05 with a drop by a factor of 0.2 at iterations {32K,48K}. We use a weight de-
cay of 0.0001. We use the standard data augmentation of RandomCrop(32, padding=4) and
RandomHorizontalFlip.
```

## Optimizers
- Adam with cosine learning rate decay and initial learning rate of 0.001.