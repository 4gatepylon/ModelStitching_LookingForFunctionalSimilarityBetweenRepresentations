# Resnet
This is where we will house all our resnet stitching code for Cifar. It will have a nice modular, declarative structure so that we can add functionality for Imagenet, etcetera.

# Goals for March 1st
1. Have experimental setup for ALL the ResNets Rumen sent me (plus ResNet9 from FFCV: try to add to Rumen's code)
2. Have modularized code
3. Have results for a subset of the ResNets stitches' plus a modular mapping interpretation
4. We will only support block-level stitching and ignore the first and last layer (unless we have extra time)

Just remember to avoid Autocasting since it can cause numerical issues and the speed should be sufficient just with regular FFCV. It is important that we finish this functionality as soon as possible (ideally the weekend, so that I can run my first batch of experiments over it and then debug) because I want to use this as a kicking stone to move on to ViT and less understood models. I need some infrastructure enable myself to read papers and do "theory" later.

# Hooks are useful for getting intermediate outputs/setting inputs, but not stitching
Read:
- https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6/
- https://discuss.pytorch.org/t/how-to-give-input-to-an-intermediate-layer/130685

# Model abstraction using state dict
So as to enable stitching that is backwards compatible with previous pretrained models, we are going to try and create a tool to
load state dicts from arbitrary models and then create a new stitchable model (where the only difference is that the new model
enables you to input or output at intermediate layers).

