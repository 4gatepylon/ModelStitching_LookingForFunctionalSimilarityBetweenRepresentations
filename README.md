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

# Febuary 23rd Update: Example Result I Got On ResNets After 1 Epoch of Stitching

## ResNet18_sim_tensor.pt
<pre>
[0.   0.   0.   0.   0.   0.  ]
[0.   0.85 0.82 0.31 0.22 0.  ]
[0.   0.29 0.87 0.52 0.49 0.  ]
[0.   0.13 0.2  0.76 0.71 0.  ]
[0.   0.11 0.12 0.19 0.9  0.  ]
[0.   0.   0.   0.   0.   0.  ]
</pre>
## ResNet18_rand_sim_tensor.pt
<pre>
[0.   0.   0.   0.   0.   0.  ]
[0.   0.22 0.24 0.2  0.2  0.  ]
[0.   0.3  0.22 0.17 0.26 0.  ]
[0.   0.14 0.2  0.11 0.18 0.  ]
[0.   0.21 0.24 0.23 0.28 0.  ]
[0.   0.   0.   0.   0.   0.  ]
</pre>
## ResNet18_debias_sim_tensor.pt
<pre>
[ 0.    0.    0.    0.    0.    0.  ]
[ 0.    0.63  0.57  0.11  0.02  0.  ]
[ 0.   -0.01  0.65  0.35  0.23  0.  ]
[ 0.   -0.02 -0.    0.65  0.53  0.  ]
[ 0.   -0.1  -0.12 -0.04  0.61  0.  ]
[ 0.    0.    0.    0.    0.    0.  ]
 </pre>

# Febuary 23rd Meeting Notes and Better Tables (40 epochs of stitching for each element in there)
## Notes
Something went wrong with the training. Here are some options:
- Loss (Nans), like the MLP: probably just remove autocasting and gradient scaling
- Rumen has a good set of hyperparameters, ask for them to train a better ResNet18: good accuracy is >= 95%
- Create some machinery for better testing (etc)
- Look at the casting issues
- ResNet18 to ResNet50 (like Yamini's ResNet20-ResNet110 but smaller): look at Rumen's code and then find something ResNet50
- Look into ViT AFTER we do ResNets

## ResNet18_sim_tensor.pt
<pre>
[0.   0.   0.   0.   0.   0.  ]
[0.   0.9  0.87 0.7  0.53 0.  ]
[0.   0.32 0.9  0.83 0.59 0.  ]
[0.   0.13 0.1  0.1  0.1  0.  ]
[0.   0.1  0.1  0.1  0.1  0.  ]
[0.   0.   0.   0.   0.   0.  ]
</pre>

## ResNet18_rand_sim_tensor.pt
<pre>
[0.   0.   0.   0.   0.   0.  ]
[0.   0.76 0.7  0.4  0.3  0.  ]
[0.   0.69 0.62 0.48 0.27 0.  ]
[0.   0.43 0.1  0.1  0.3  0.  ]
[0.   0.1  0.1  0.1  0.27 0.  ]
[0.   0.   0.   0.   0.   0.  ]
</pre>

## ResNet18_debias_sim_tensor.pt
<pre>
[ 0.    0.    0.    0.    0.    0.  ]
[ 0.    0.14  0.16  0.3   0.23  0.  ]
[ 0.   -0.37  0.28  0.36  0.33  0.  ]
[ 0.   -0.3   0.    0.   -0.2   0.  ]
[ 0.    0.    0.    0.   -0.17  0.  ]
[ 0.    0.    0.    0.    0.    0.  ]
</pre>

# Goals for March 1st
1. Have experimental setup for ALL the ResNets Rumen sent me (plus ResNet9 from FFCV: try to add to Rumen's code)
2. Have modularized code
3. Have results for a subset of the ResNets stitches' plus a modular mapping interpretation

Just remember to avoid Autocasting since it can cause numerical issues and the speed should be sufficient just with regular FFCV. It is important that we finish this functionality as soon as possible (ideally the weekend, so that I can run my first batch of experiments over it and then debug) because I want to use this as a kicking stone to move on to ViT and less understood models. I need some infrastructure enable myself to read papers and do "theory" later.

Note that these goals are in more detail in the `README.md` in the `resnet` folder which is going to be our new repository for...

# March 2nd Status

Models: Resnet18 and Resnet34 have high accuracies: around 95% each.

Rumen says that it is not a terrible idea to try out stitching with the cached outputs as our "labels" instead of doing full backprop. Try with mean squared error. (This is lower priority than debugging and small networks).

Try with very small resnets (i.e. 1,1,1,1).

Implement interpretation algorithms.

Look into scale (how you normalize and also potentially non-linearity, but that comes later.) This is only later if we use more differnet networks.

Also average and find some variance/stdev between the depth dimension of the experimental grids. Run experiments in parallel and overnight for speed.

# March 9th Status
- Trained resnet `1111`, `1112`, ... up to resnet `2222` (which is resnet 18).
- For every pair, tried stitching across block-sets (there are `16` networks total, meaning that there are `120` pairs, the layers are on average `8` meaning that we have `960` stitches, so it might take a little while... especially since we also do `4` combinations per out of either rand or normal, so this is a trully monstrous experiment of `3840` stitches total, each for three epochs yielding `11520` total epochs, at around `30` sec per epoch maximum which is around `96` hours of training time if we run this constantly... It'll be better to instead have 16 or so )
- I did not save the stitches because they are fast to train and I found out that saving every stitch just from Resnet18 to Resnet34 takes up 15 gigabytes per experiment. We also don't store log files for the same reason...
- Note I'm racking up a lot of technical debt (my code is seriously ugly and super repetitive: it would be nice to simplify and clean up this nonsense once I have some results I'm happy with as a starter or once I need to)
- Might want to look at "latest layer than can stitch" instead of maximum
- Don't forget pruning
- Don't forget Cifar100

From meeting
- Look at fixed points papers
- Try lowering the width/depth of the layers
- Try without residuals
- If you plug in the dataset and compare the activates with the activations you'd get through the stitch see what happens (push it through the network)... use some similarity metric to see if they are the same (idk mean squared?): alternative is the stitch found a different way of providing a representation the other network can use even though it's not the same
- Read about salieny maps (layerwise? gradient saliency?): this stems from the idea of creating an image that somehow conveys what the network is extracting and using stitches to somehow get that'
- Pruning interesting?

Things to do
- __Most important experiment: Autoencoder from stitch__
  - Train resnet on dataset
  - Take some prefix of the resnet
  - Cache the outputs
  - Cache the outputs of the layer coming after
  - Train stitch to do what that resnet does? Basically do mean squared error or L1 loss.
  - Interpretation: can stitch re-create dataset from the activations?
Experiment to do:
- If it's easy to reproduce what does that mean?
- If it's hard?

- Compare vanilla stitch with autoencoder stitch?

If we have more time do these too:
- Same thing no residuals (it's ok if the accuracy is around 80+)
- Look into VGG but probably hard since not so modular
- Try removing the layers (not just the blocks): less important

Goal: understand why what is happening is happening.