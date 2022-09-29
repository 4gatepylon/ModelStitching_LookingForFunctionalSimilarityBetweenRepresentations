# TODO
Put everything into the experiments folder.

Add all the sanity tests from `mega.py` as unit tests.

# Overview
Here we have a relatively modular and well-tested stitching library. It is the `stitching` module that you import
things from by doing `from stitching.abc.py import xyz` for some `abc`  and `xyz` of your choice. To unit test this library simply run `python -m unittest discover -v -s . -p "*.py"` (probably make an alias, on my stystem it's `stest`) from within the `stitching` directory.

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
- File Manager (utility to save models in an orderly fashion)
- Rep Dataset so that we can do model stitching with similarity loss (or whatever)

# TODOs
- Implement the above
- checkValid should be renamed to something like assertRep and use asserts insofar as is possible so that we can optimize it away with the `-O` flag. Alternatively, we could just insert asserts everywhere. The easiest thing woudld be to create a function called `checkInternals` that returns a boolean (dummy) and that internally has a ton of asserts. You would use it by asserting it. That way we can `-O` it away. For this we should implement a `CheckInternals` superclass. Apprently multiple inheritance is supported (check https://pythonprogramminglanguage.com/multiple-inheritance/) so that may be the pattern we want to use.
- prefix and suffix should be made to 
- we should have a dummy `Resnet` subclass that just wraps arbitrary modules so as to allow type annotations (type annotations accept all possible subclasses)
- should probably set the `__all__` properly for all these classes

From my old code for `experiment.py`:

```
# TODO
        # 3. Get LIST of tables of stitches [vanilla_table, sims_table]
        # 4. Get LIST of tables table of stitched networks [vanilla_table, sims_table]
        # (remember to get idx2label for ease of debugging)
        # 5. Use table mapping to train the stitched networks, getting their sims
        # 6. Use table mapping to get the mean2
        # 7. Save sims to a file (or call visualizer)
        # 8. Save mean2 to a file (or call visualizer)
        # 9. Select the stitches that stitch into into conv1
        #    For each of these stitches sample N = 5 images (or some other constant)
        #    For each of these images input them into the prefix, then into the stitch
        #    Then use our image functionality to un-normalize it. Store that image into
        #    A file. Next to it store the image that was input (this will help us
        #    visualize whether the stitch is doing any funny business).
```