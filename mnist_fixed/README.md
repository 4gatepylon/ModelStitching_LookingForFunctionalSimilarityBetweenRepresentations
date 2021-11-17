# Running
You are encouraged to run `python3 -O main.py` (with the `-O` to disable assertions and enable "optimizations"). For debugging
make sure to run without `-O`.

Immediate next steps include
1. Ensuring that on stitch, the two networks are frozen (through assertions, etcetera)
2. Start doing MNIST long tests with some form of resnet (or at least build the infrastructure); MAKE SURE TO FULLY AUTOMATE so that we can compare FOR DIFFERING AMOUNTS OF EPOCHS AND OTHER STUFF TOO (mnist short requires you to do your experiments by hand for the differing amounts of epochs, which is really annoying)

## Running on Supercloud
1. Make sure to copy data with `scp` or something like `transfer.sh` (using `wget` on the upload URL)
2. Make sure to `git pull` the latest master (or whatever)
3. Run an interactive session of around one day with `LLsub -i -s 20 -g volta:1` (this is half a node, but for MNIST we don't need that much)
4. Get info about what you are getting with `LLstat` Running status is `R`, while `PD` is pending, etc...
5. Make sure before that you're running on an anaconda module (or some other python3 compatible one for pytorch). Use `module --help` to get help, but basically you can use `module avail` and `module load ...` and then `module list` to see what you can get, load something, then see what you've loaded.