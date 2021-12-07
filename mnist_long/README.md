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
PLEASE LEARN TO USE BATCH