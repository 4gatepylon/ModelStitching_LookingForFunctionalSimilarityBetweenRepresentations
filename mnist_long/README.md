# Things To Do
1. Debug! Get the experiments to run on supercloud!
2. Stitch into and out of a single layer (i.e. an "insertion" stitch instead of a swap)
3. Sanity tests for frozen layers and unfrozen layers
4. Documentation and library simplifcation for the model generation and stitch generation (read their code and see what they do)
5. Find a way to analyze based on the accuracies
6. Autogenerate examples from higher level information
7. Start writing up Latex results... figure out how to visualize this nicely...

# Running on Supercloud
More info the fixed folder.

Make sure to copy data with `scp` or something like `transfer.sh` (using `wget` on the upload URL)

Run

```
module load anaconda/2021a && module load cuda/11.2 && LLsub -i -s 20 -g volta:1
```

Then

```
cd mnist_long && rm -rf experiments && python3 main.py
```

# Important
- Use batch for overnight runs instead of just interactive
- Use all your 16 max possible usable GPUs

# Ideas for the future
- Constrained optimization or constrained stitch
- Linearity as being constrained to subsets/sub-dimensions (i.e. this defines the "format")
- What happens in you vary random initialization?
- Deal with different input and output shapes that is not just hacky