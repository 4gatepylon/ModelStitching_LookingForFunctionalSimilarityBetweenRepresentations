# Plato
Contrastive Learning SuperUROP Fall 2021 - Spring 2022

Right now most of the code is in `mnist_fixed.py` which simply does stitching for the MNIST tutorial network from pytorch.

You are encouraged to run `python3 -O mnist_fixed.py` (with the `-O` to disable assertions and enable "optimizations"). For debugging
make sure to run without `-O`.

Immediate next steps include
1. Ensuring that on stitch, the two networks are frozen (through assertions, etcetera)
2. Measuring whether the stitch is effective (please use the paper)
3. Tensorboard plots so we can compare behavior over time (may want to do log plots, or something, since even tiny networks reach high accuracies)
4. Start thinking about how to expand this to longer networks (perhaps that will begin a new file, `mnist_long.py` or something like that); I think before that, though, it may be more important to enable YAML hyperparameter searching, ensure that we can run on SuperCloud (may involve one or two python scripts), adding baselines, and shorten widths to do more reasonable testing.