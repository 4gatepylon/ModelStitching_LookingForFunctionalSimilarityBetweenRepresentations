# Running
You are encouraged to run `python3 -O main.py` (with the `-O` to disable assertions and enable "optimizations"). For debugging
make sure to run without `-O`.

Immediate next steps include
1. Ensuring that on stitch, the two networks are frozen (through assertions, etcetera)
2. Measuring whether the stitch is effective (please use the paper)
3. Tensorboard plots so we can compare behavior over time (may want to do log plots, or something, since even tiny networks reach high accuracies), this also includes utility logging, etcetera
4. Start thinking about how to expand this to longer networks (perhaps that will begin a new file, `mnist_long.py` or something like that); I think before that, though, it may be more important to enable YAML hyperparameter searching, ensure that we can run on SuperCloud (may involve one or two python scripts), adding baselines, and shorten widths to do more reasonable testing.

## Running on Supercloud
1. Make sure to copy data with `scp` or something like `transfer.sh` (using `wget` on the upload URL)
2. Make sure to `git pull` the latest master (or whatever)
3. Run an interactive session of around one day with `LLsub -i -s 20 -g volta:1` (this is half a node, but for MNIST we don't need that much)
4. Get info about what you are getting with `LLstat` Running status is `R`, while `PD` is pending, etc...
5. Make sure before that you're running on an anaconda module (or some other python3 compatible one for pytorch). Use `module --help` to get help, but basically you can use `module avail` and `module load ...` and then `module list` to see what you can get, load something, then see what you've loaded.