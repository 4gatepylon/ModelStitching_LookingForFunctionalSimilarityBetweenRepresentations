from examples_cnn import NET_3_2 as C32

from hyperparams import (
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_TEST_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_LR_EXP_DROP,
    DEFAULT_EPOCHS_OG,
    DEFAULT_EPOCHS_STITCH,
    NUM_EXPERIMENTS,
    NUM_STITCH_EXPERIMENTS,
    DOWNLOAD_DATASET,
)

from net import (
    Net,
    get_stitches,
    StitchedForward,   
)

# This file is going to do basically what main does, but only sanity checks
# Sanity checks are going to be simple. We illustrate one sanity check below. We will do
# more later (involving two-way stitching, multiple pivots, more statistics.) Moreover,
# we will want to do this sanity check for MULTIPLE ARCHITECTURES!

# NOTE
# The sanity check will be run on the 3-layer CNN
# We will change our code to use regular stochastic gradient descent

# SELF-STITCH SANITY CHECK (One way stitch; Only for CNNs; Only one architecture):
# 0. Fix the model architecture.
# 1. Train it n times into M_0, M_1, ..., M_n-1
# 2. Choose a random model to be the "pivot" P
#    (this will not work if they don't stitch, but whatever)
# 3. For all i (i.e. all models, including when P = M_i) 
#      For all layers j before the last layer of P
#        For all layers k after the first layer M_i
#           Train Stitch(P[j: OUTPUT], M_i[k: INPUT]) INTO M_i.
#           That is to say, <P[j] -> St -> M_i[k]<.
#           This stitch "St" will be stored as Stitches[0][i][j][k] to signify
#           that it is from model 0 (i.e. the Pivot) to model i from layer j
#           into layer k. Moreover, store the stitching penalty, which is
#           Acc(min(M_i, P)) - Acc(Stitched(M_i, P)), into penalties[0][i][j][k].
# 
# 4. Let AvgPenalty[j][k] = Avg(penalties[0][:][j][k]). This is the average penalty from the
#    output of layer j, inputting into layer k. We now have a sense of how much penalty we should
#    expect between layer j and layer k.
# 5. FOR CNNs do SANITY CHECKS:
#    DEFINE epsilon: A small number, signifiying that stitching penalty between equivalent
#      architectures (at corresponding layers) should be close to zero.
#    DEFINE epsilon0: a small number, signifiying that stitching penalty between the same
#      literal instance and architecture (at the same layer) should be close to zero. This should
#      be smaller than epsilon.
#    DEFINE epsilon_var_l1: a small number, signifying that stitching corresponding layers should
#      consistently be acccurate, not just on average. This is for the average distance from the average
#      (i.e. Exp(|x_i - Exp[x]|)).
#    DEFINE epsilon_var_l2: The same as for l_1, but for the variance: Exp[(x_i - Exp[x])**2].
#    DEFINE delta: some big number signifying that different layers should NOT be stitchable on the
#      same network.
#    ...
#    For all j
#      ASSERT AvgPenalty[j][j+1] <= epsilon:
#        "self-stitch 1:1 works."
#      ASSERT penalties[0][0][j][j+1] <= epsilon0:
#        "self-stitch 1:1 same instance works best."
#      ASSERT L2_VarPenalty[j][j+1] <= epsilon_var_l2 AND L1_VarPenalty[j][j] <= epsilon_var_l1:
#        "self-stitch 1:1 should consistently achieve accurate results."
#      ...
#      For all k != j
#        ASSERT AvgPenalty[j][k] >= delta.
# 6. Create visualizations:
#   VISUALIZE penalties[0][:][j][k] (one plot per j, k pair).
#   VISUALIZE AvgPenalty by creating a heat-map grid with each row or column being
#     j and k - 1. Very importantly, note that all the j, k elements go to -> j, k - 1.
#     This allows us to compare the representations as we wanted to do to begin with.

def train_n(train_func, n):
    models = [None] * n
    accs = [0] * n
    for _ in range(n):
        models[i], accs[i] = train_func()
    return models

def stitch_all(stitch_func, valid_st_func, models, accs, num_layers,  pivot_idx=0):
    # penalties[n][num_layers * num_layers]
    # index penalties[i][j][k] as penalties[i][j*num_layers + k]
    # assuming k < num_layers and j < num_layers
    stitches = [None] * n
    penalties = [None] * n
    for i in range(n):
        penalties[i] = [None] * (num_layers * num_layers)
        stitches[i] = [None] * (num_layers * num_layers)
    
    pivot = models[pivot_idx]
    for i in range(len(models)):
        compared = models[i]
        # exclude the last layer as outputting
        # do not "output" the image
        for layer_1_outputting in range(1, num_layers):
            # include the last layer as inputting
            # do not input into the first layer
            for layer_2_inputting in range(2, num_layers+1):
                if valid_st_func(layer_1_outputting, layer_2_inputting):
                    # we are comparing layers j with k by inputting j's output
                    # into k+1's input (expecting k's output) through a stitch
                    # this is a little different from the docs above, so keep it in mind
                    j = layer_1_outputting
                    k = layer_2_inputting - 1
                    stitched, st_acc = stitch_func(pivot, compared, layer_1_outputting, layer_2_inputting)
                    stitches[i][j * num_layers + k] = stitched

                    # note that we can recover the st_accs using accs, so no need to store
                    penalties[i][j * num_layers + k] = min(accs[pivot_idx], accs[i]) - st_acc
    
    # note that for the sanity checks it may not be necessary to store
    # stitches and instead we may prefer to delete to lower memory consumption
    return stitches, penalties

# find L1, L2, etcetera...
def derive_penalty_statistics(penalties):
    avg, L1, L2 = None, None, None
    return avg, L1, L2 # TODO

def make_sanity_checks(
    num_layers,
    penalties, avg_penalties, L1_penalties, L2_penalties, 
    epsilon0, epsilon1, epsilonl1, epsilonl2, delta):
    # make sure that models were self-stitchable for single models
    for n in range(penalties):
        for i in range(num_layers):
            assert penalties[n][i][i] <= epsilon0
    
    # make sure that models were self-stitchable for a given architecture
    # and that this was the case consistently
    for i in range(num_layers):
        assert avg_penalties[i][i] <= epsilon
        assert L1_penalties[i][i] <= epsilonl1
        assert L2_penalties[i][i] <= epsilonl2
    
    # make sure that layers that really should be different were not similar
    for i in range(num_layers):
        for j in range(0, i):
            assert avg_penalties[i][j] >= delta
        for j in range(i+1, num_layers):
            assert avg_penalties[i][j] >= delta

# TODO visualizations
# 1. Grid heatmap (of average penalties)
# 2. Depth dimension of that grid ^ (i.e. not the average, but the distribution): one per i, j pair