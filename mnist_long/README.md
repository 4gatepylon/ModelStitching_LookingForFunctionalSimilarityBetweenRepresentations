# January ASAP
`NOTE: This is the most important, we are aiming for Jan 25th (Tuesday)`
1. Finish sanity checks and test on the tiny cnn. Make sure to visualize! You can store shit in memory. Make sure it's freezing.
2. Test sanity checks on the large cnn. Make sure to visualize!
3. Investigate what the status of "main" is. It is very odd because inter-stitching is not well defined for validity.
4. Create a declarative language for stitching so I don't have to struggle with this nonsense later. Create a "compiler" of sorts for your language.

# Notes from December
For large experiments, get the experiments to run on `16 GPUs at once running out of memory`!
- https://stackoverflow.com/questions/57858433/how-to-clear-gpu-memory-after-pytorch-model-training-without-restarting-kernel
- https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html

## Immediately After ^ is done
1. Stitch into and out of a single layer (i.e. an "insertion" stitch instead of a swap)
2. Sanity tests for frozen layers and unfrozen layers
3. Documentation and library simplifcation for the model generation and stitch generation (read their code and see what they do)
4.  Find a way to analyze succes or not based on accuracies. Parse files (important).
5. `I DO NOT KNOW WHAT THIS IS` Autogenerate examples from higher level information
6. `ONGOING` Start writing up Latex results... figure out how to visualize this nicely...

# Running on Supercloud
More info the fixed folder.

Make sure to copy data with `scp` or something like `transfer.sh` (using `wget` on the upload URL)

Run

```
module load anaconda/2021a && module load cuda/11.2 && LLsub -i -s 20 -g volta:1
```

To run a batch job do (though this might not work because "permission denied"... just copy the code it's a single line)
```
./batch.sh
```

Then

```
cd mnist_long && rm -rf experiments && python3 main.py
```

# Ideas for the future
- Constrained optimization or constrained stitch
- Linearity as being constrained to subsets/sub-dimensions (i.e. this defines the "format")... This is beginning to be explored in my "outline" in google doc. I had to pause it to focus on what was highest priority.
- What happens in you vary random initialization?
- Deal with different input and output shapes that is not just hacky! This could be related to the "linearity as being constrained to subsets/sub-dimensions"