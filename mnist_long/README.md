# TODO ASAP
Get the experiments to run on `16 GPUs at once running out of memory`!

## Immediately After ^ is done
1. Stitch into and out of a single layer (i.e. an "insertion" stitch instead of a swap)
2. Sanity tests for frozen layers and unfrozen layers
3. Documentation and library simplifcation for the model generation and stitch generation (read their code and see what they do)
4. Find a way to analyze based on the accuracies
5. Autogenerate examples from higher level information
6. Start writing up Latex results... figure out how to visualize this nicely...
7. Try "swapping out" matching modules (i.e. stitch and then stitch back)

# Running on Supercloud
More info the fixed folder.

Make sure to copy data with `scp` or something like `transfer.sh` (using `wget` on the upload URL)

Run

```
module load anaconda/2021a && module load cuda/11.2 && LLsub -i -s 20 -g volta:1
```

To run a batch job do
```
module load anaconda/2021a && module load cuda/11.2 && LLsub batch.sh -s 20 -g volta:1
```

Then

```
cd mnist_long && rm -rf experiments && python3 main.py
```

# Ideas for the future
- Constrained optimization or constrained stitch
- Linearity as being constrained to subsets/sub-dimensions (i.e. this defines the "format")
- What happens in you vary random initialization?
- Deal with different input and output shapes that is not just hacky