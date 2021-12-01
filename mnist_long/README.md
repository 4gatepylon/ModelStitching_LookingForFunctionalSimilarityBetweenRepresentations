# Feature Requests
1. Seperate number of training epochs for stitches vs. for source networks
2. More efficient stitching; abandon stitch mode and instead adopt a superior method of indexing layers
3. Sanity tests for frozen layers and unfrozen layers
4. Baselines (random network, random stitch)
5. Ability to give experiments concise and semantically relevant names and generate a readme inside them.
6. Ability to disable RNG with seeds + sanity check for RNG + Ability
7. Ability to run and compare multiple experiments (i.e. one experiment with seeds, one without)
8. `TODO`

# Running on Supercloud
1. Make sure to copy data with `scp` or something like `transfer.sh` (using `wget` on the upload URL)
2. Make sure to `git pull` the latest master (or whatever)
3. Run an interactive session of around one day with `LLsub -i -s 20 -g volta:1` (this is half a node, but for MNIST we don't need that much)
4. Get info about what you are getting with `LLstat` Running status is `R`, while `PD` is pending, etc...
5. Make sure before that you're running on an anaconda module (or some other python3 compatible one for pytorch). Use `module --help` to get help, but basically you can use `module avail` and `module load ...` and then `module list` to see what you can get, load something, then see what you've loaded.
6. If you've been testing and have BS experiments you can do `rm -rf experiment-tensorboard-*` or `rm-rf experiment-tensorboard-<date>-*`.
7. To display your tensorboard first download as so: `scp -r ahernandez@txe1-login.mit.edu:/home/gridsan/ahernandez/git/Plato/mnist_fixed/experiment-tensorboard-<download label> experiment-tensorboard-<label>` and then in the same location `tensorboard --logdir=experiment-tensorboard-<label>`. Ideally use a simplified and semantically relevent `<label>` as opposed to the original (probably datetime) `<download label>`.