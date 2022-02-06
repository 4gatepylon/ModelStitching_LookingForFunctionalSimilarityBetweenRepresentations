# Plato
Contrastive Learning SuperUROP Fall 2021 - Spring 2022. The most recent code is in the long folder (used to be mnist_long) but it may no longer be totally functional. Big refactors are currently ongoing.

Previously this was meant to be able to run both locally and on supercloud, however, local runs are no longer fully supported, and you may run into issues.

## Status
Right now we are working on sanity testing with CIFAR-10. The goal is to show that stitching can work and see whether interesting behavior comes out. The immediate next steps are these:
1. Enable FFCV in SuperCloud and confirm (using interactive prompt) that we can run CIFAR-10 trainings very quickly. This will require good `conda` package management.
2. Train experiments on CIFAR-10: `ResNets` (based on shape) as well as `CNNs` and `MLPs` from my library. We are NOT going to be looking at attention. This will result in `similarity tables` from every layer to every other (we can put zero for layers that are not stitchable). This will require me to `follow up with Yamini` for her results with the ResNets. This will require the experiments to also have baselines.
3. Look into distillation (follow up with Yamini) and set up the stithing language if initial sanity tests with FFCV are good.

## How To Use
Be on SuperCloud or some other GPU-enabled environment. Install anaconda. Pull the latest verion of Plato. You can connect using `ssh ahernandez@txe1-login.mit.edu` or an alias (i.e. `supercloud`).

If you are not using FFCV, you can run `module load anaconda/2022a && module load cuda/11.3` to get faster-importing modules for machine learning. Otherwise, you can simply use the FFCV environment we will create.

Run the FFCV environment setup: `conda create -y -n ffcv python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge`. This will create in the anaconda virtual environment that does NOT inherent from your base environment. This environment is basically a folder that Anaconda install libraries into and from which Anaconda reads only while you are in it, so to speak. This should be long-lived (i.e. you won't have to run this every single time). You can check if it exists by using `conda env list`. It is possible to install these environments in your own folder somewhere, but we do this for simplicity. Conda/env resources are linked below:
- https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533
- https://www.anaconda.com/blog/understanding-conda-and-pip
- https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html
- https://stackoverflow.com/questions/52433517/do-newly-created-conda-envs-inherit-all-packages-from-the-base-env

Start the FFCV environment and install ffcv: `conda activate ffcv`, `pip install ffcv`. Make sure to run `conda deactivate` when you are done. Note that the non-login instances on Supercloud are not connected to the internet. That is OK as long as you do this setup on a login instance and then activate it on a regular GPU-enabled instance. Also, if you wish to sanity test that you are in the right environment use `which python`. You should not need to use the python3 command, and if using python invokes some other version (i.e. not 3.9) then something has gone wrong.

Run your SuperCloud job.
- Interactive: `LLsub -i -s 20 -g volta:1`. Use this to test or for short training jobs.
- Batch: `LLsub batch.sh -s 20 -g volta:1`. If you are running batch make sure that `batch.sh` exists and is correct. It should have one line with `#!/bin/bash` and one with the command you wish to run (i.e. `python -O main.py > log.txt`). Remember to use `-O` to tell python to avoid using asserts. Do NOT write code with asserts in long loops (instead, make a function and assert on the output of the function) so that the loop is not traversed regardless of whether you pass `-O`.

After your job is done, you can copy files back (logs, visualization, pre-trained weights) to local using scp:
- Copy a file: `scp ahernandez@txe1-login.mit.edu:/home/gridsan/ahernandez/git/Plato/.../myfile mynewfile` (or a directory instead of mynewfile if you want to keep the name)
- Copy a folder: `scp -r ahernandez@txe1-login.mit.edu:/home/gridsan/ahernandez/git/Plato/... mynewfolder`

In the future we will add instructions for how to run on multiple GPUs.