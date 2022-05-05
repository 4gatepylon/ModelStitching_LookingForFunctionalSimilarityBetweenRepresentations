# Sanity
The goal here is to sanity test things because I've observed behavior that is 100% busted:
1. If you stitch from one model to itself vs. one model to a copy it behaves differently
2. If you store and load stitches then suddenly their accuracy on the downstream task changes dramatically
3. If you evaluate multiple times in a row you get different accuracies.
4. If you use different batch sizes, the behavior can vary.

This is all regadless of whether you use FFCV or whatever. Something is dreadfully wrong and I'm not sure what it is
but we need to isolate what is causing these issues so that we can ensure 100% determinism and then do our stitches in an
environment that we can believe in.

What we want to try for this situation is:
1. Try deepcopy: this is meant to be a workaround/comparison of model loading. Does behavior change if we copy in memory instead of initializing and then loading state dict?
2. Make sure stitches are the same: we should assert that the weights of the stitches do not change from store to load.
3. Make sure no data shuffling: it is possible that I forgot to leave out a keyword argument or something like that to disable shuffling (which may be enabled by default).
4. Remove abstractions: make the code as simple as possible by removing the stitched network and just using stitches in the eval loop (note that we already have models pretrained so it should be OK to just use them... though at this point we have to be careful even of that).
5. Make sure everything is evaluated (esp. for the buffers): basically, if we are using `model.train()` anywhere there is a chance that the buffers (i.e. for batch-norm) are still evaluating which means that when we do partial application of the neural network some are left untouched, others aren't, and overall behavior is wierd. I don't understand well how this could impact us, but I can see that it wouldn't be good.
6. Try Little Models (and CPUs): basically a good heuristic to try and quickly isolate what's going on.
7. Look at floating point issues, grad scaler, and so on so forth: basically, it's possible that because our models are stored as half (at least I think they are, but I'm not sure) there is wierd shit going on. Not sure what the effect would be other than simple numerical mistakes (and I don't know how that would cause the triangle of death or any other wierd behavior) but generally, it may be bad practice for us to let it be that way.
8. Loading problem: just try to load and not load, cuda not cuda, linear layer: just make sure that load/store works.