# Running
You are encouraged to run `python3 -O main.py` (with the `-O` to disable assertions and enable "optimizations"). For debugging
make sure to run without `-O`.

Immediate next steps include
1. Ensuring that on stitch, the two networks are frozen (through assertions, etcetera) `TODO`
2. Start doing MNIST long tests with some form of resnet (or at least build the infrastructure); MAKE SURE TO FULLY AUTOMATE so that we can compare FOR DIFFERING AMOUNTS OF EPOCHS AND OTHER STUFF TOO (mnist short requires you to do your experiments by hand for the differing amounts of epochs, which is really annoying) `TODO FEATURE REQUESTS` (note that we would like for this to run overnight or when I am not working, otherwise I'm gonna be way too inefficient) (also we will want a more efficient way to do the stitch that doesn't involve a million if statements)

## Running on Supercloud
1. Make sure to copy data with `scp` or something like `transfer.sh` (using `wget` on the upload URL)
2. Make sure to `git pull` the latest master (or whatever)
3. Run an interactive session of around one day with `LLsub -i -s 20 -g volta:1` (this is half a node, but for MNIST we don't need that much)
4. Get info about what you are getting with `LLstat` Running status is `R`, while `PD` is pending, etc...
5. Make sure before that you're running on an anaconda module (or some other python3 compatible one for pytorch). Use `module --help` to get help, but basically you can use `module avail` and `module load ...` and then `module list` to see what you can get, load something, then see what you've loaded.
6. If you've been testing and have BS experiments you can do `rm -rf experiment-tensorboard-*` or `rm-rf experiment-tensorboard-<date>-*`.
7. To display your tensorboard first download as so: `scp -r ahernandez@txe1-login.mit.edu:/home/gridsan/ahernandez/git/Plato/mnist_fixed/experiment-tensorboard-<download label> experiment-tensorboard-<label>` and then in the same location `tensorboard --logdir=experiment-tensorboard-<label>`. Ideally use a simplified and semantically relevent `<label>` as opposed to the original (probably datetime) `<download label>`.