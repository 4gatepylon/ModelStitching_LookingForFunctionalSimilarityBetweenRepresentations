# Resnet
This is where we will house all our resnet stitching code for Cifar. It will have a nice modular, declarative structure so that we can add functionality for Imagenet, etcetera.

# Goals for March 1st
1. Have experimental setup for ALL the ResNets Rumen sent me (plus ResNet9 from FFCV: try to add to Rumen's code)
2. Have modularized code
3. Have results for a subset of the ResNets stitches' plus a modular mapping interpretation
4. We will only support block-level stitching and ignore the first and last layer (unless we have extra time)

Just remember to avoid Autocasting since it can cause numerical issues and the speed should be sufficient just with regular FFCV. It is important that we finish this functionality as soon as possible (ideally the weekend, so that I can run my first batch of experiments over it and then debug) because I want to use this as a kicking stone to move on to ViT and less understood models. I need some infrastructure enable myself to read papers and do "theory" later.

# Important Note!
Remember that we want to work smart, not hard! We may be able to avoid a lot of work refactoring the models for stitching by using a forward hook as so: https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6/. The alternative is to modify the model code directory, put everything in lists, and iterate...

Immediate next step should be to try forward hook on resnet9 to see if it's a good alternative.

We may also want to ask Rumen for hyperparameters (lr, scheduling, etcetera) if we are going to be re-training these resnets due to any modifications to them (changing the names of the layers, etcetrera), which is likely to happen. The way I see it there are two main ways to do this:
1. Add dummy identity layers that we can hook onto with a standardized format OR simply standardize the nams of the block outputs
2. Create lists of layers and go from one index to another