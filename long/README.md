# Note
This readme is somewhat deprecated.

# What is left for sanity
Use visualizations, re-read the paper and compare ideas.

# Helpful tips
To create a batch run for sanity tests run `module load anaconda/2021a && module load cuda/11.2 && LLsub sanity_batch.sh -s 20 -g volta:1`.

Make sure that sanity_batch has the right file: `mnist_sanity` or `cifar_sanity` depending on which one you want to run.

Note that the CIFAR sanity has less testing and is optimized for runtime, while the mnist sanity runnable is meant for testing (and thus has assertions intersperesed everywhere). Make sure to run with `python3 -O (mnist_sanity | cifar_sanity).py` so that you can avoid assertions if you want to do so (though you will still run their loops in many cases for mnist since we did not optimize for that).

You can pipe into a file with `python3 mything.py > myfile.txt`.

You can copy files by `scp ahernandez@txe1-login.mit.edu:/home/gridsan/ahernandez/git/Plato/mnist_long/myfile mynewfile`. You can use `-r` to copy a directory by ommitting the second file.

# Things to consider doing:
For large experiments, get the experiments to run on `16 GPUs at once running out of memory`!
- https://stackoverflow.com/questions/57858433/how-to-clear-gpu-memory-after-pytorch-model-training-without-restarting-kernel
- https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html

Stitch into and out of a single layer (i.e. an "insertion" stitch instead of a swap). Alternatively, try two-way stitching.

Documentation and library simplifcation for the model generation and stitch generation (read their code and see what they do)

Writeup latex results and interpret behavior.

# Running Interactive
```
module load anaconda/2021a && module load cuda/11.2 && LLsub -i -s 20 -g volta:1
```

# Ideas for the future
- Constrained optimization or constrained stitch
- Linearity as being constrained to subsets/sub-dimensions (i.e. this defines the "format")... This is beginning to be explored in my "outline" in google doc. I had to pause it to focus on what was highest priority.
- What happens if you vary random initialization?
- Deal with different input and output shapes that is not just hacky! This could be related to the "linearity as being constrained to subsets/sub-dimensions"

# If you ever end up using deprecated_main...

You want to launch two batch jobs, one for each mode (0 and 1):
```
#!/bin/bash
python3 main.py --MODE=0
```