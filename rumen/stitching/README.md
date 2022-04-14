# Overview
Here we have a relatively modular and well-tested stitching library. It is the `stitching` module that you import
things from by doing `from stitching.abc.py import xyz` for some `abc`  and `xyz` of your choice. To unit test this library simply run `python -m unittest discover -v -s . -p "*.py"` from within the `stitching` directory.

Where possible, I created classes (objects) to execute shared functionality and maintain shared state. All objects that group methods, and especially that maintain state, are unit tested. For example, you will find a `LayerLabel` class that basically creates utility for us to tell `ResNets` when to prematurely stop, it tells us what pairs of layers are being compared, etcetera. There is in this file a `TestLayerLabel` that unittests all the non-trivial functions of the class. In general, every single class will have a corresponding unit tester in the same file. I try to seperate classes to one per file to avoid bloat.

# Layer Label
1. Creates an object that knows what layer in what network it corresponds to
2. Can be transformed into an index, or, given a network, gotten from an index.
3. Can create a table (just a list of lists) given two networks, that captures all the pairs of labels possible between those two networks. This table's entries will be some closure called the "constructor" which is passed in. The constructor will normally be something like creating a stitch, training a stitch, etcetera.

# Net Trainer: Not Yet Implemented
Given a network and some loss, this is able to train that network to high accuracy (or at least as high as we can hope to get otherwise). This should store state for hyperparameters, etcetera, and generally wrap the functionality that Rumen gave me in the training loop.

# Net Generator: Not Yet Implemented
A class that creates "pretrained" networks for stitching. It only supports basic pretraining using default randomized layers and loading cached resnets from a given folder.

# Stitch Generator: Not Yet Implemented
A class that encapsulates and unit tests methods to create stitches and to a lesser extent `StitchedResnet` objects.

# StitchedResNet: Not Yet Implemented
Simply a `nn.Module` that enables us to, given two networks and corresponding labels, take the output from the layer corrresponding to the first label in the first network and then put that through a given stitch and put it into the input of a layer corresponding to the second label in the second network.

# Miscellaneous: Not Yet Implemented
We are going to need functions to
- Visualize heatmaps
- Visualize images
- Do mean2 error and other such metrics
- Generate the sequence of networks relevant for an experiment (i.e. some sort of experiment class)
- Manage files for an experiment (some sort of experiment file manager class)
- Selection of dataloaders from either default Pytorch ones for CNNs or FFCV's ones.