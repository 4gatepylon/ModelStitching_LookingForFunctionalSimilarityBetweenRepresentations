import os
import argparse
from tkinter.messagebox import NO
from unittest.mock import DEFAULT
import torch
import torch.optim as optim

from datetime import datetime
from torch.optim.lr_scheduler import StepLR

from examples_cnn import (
    NET_3_2 as C32,
    NET_3_2_TO_NET_3_2_STITCHES as C32T32,
)
from examples_tinycnn import (
    NET_CTINY as CT,
    NET_CTINY_TO_NET_CTINY_STITCHES as CT2T,
)

from hyperparams import (
    # NOTE these are default form the Pytorch MNIST example
    # (and so are train and test batch size)
    DEFAULT_LR,
    DEFAULT_LR_EXP_DROP,

    # NOTE these may require some tweaking
    DEFAULT_EPOCHS_OG,
    DEFAULT_EPOCHS_STITCH,
)

from net import (
    Net,
    get_stitches,
    StitchedForward,   
)

from training import (
    init,
    train1epoch,
    test,
)

DEFAULT_SANITY_INSTANCES = 5

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
#

def train_og(model, device, train_loader, test_loader, epochs=DEFAULT_EPOCHS_OG):        
    # NOTE we may want to use vanilla gradient descent in the future, but this should be fine (it's default).
    # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    # https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
    optimizer = optim.SGD(model.parameters(), lr=DEFAULT_LR)
    scheduler = StepLR(optimizer, step_size=1, gamma=DEFAULT_LR_EXP_DROP)

    for epoch in range(1, epochs):
        print("Training og epoch {}".format(epoch))
        train1epoch(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()
    train1epoch(model, device, train_loader, optimizer)
    _, test_acc = test(model, device, test_loader)
    return test_acc

# NOTE that we use reciever format
def train_stitch(model1, model2,
    layer1_idx, layer2_idx,
    stitch,
    device, train_loader, test_loader, epochs):
    stitched_model = StitchedForward(model1, model2, stitch, layer1_idx, layer2_idx)

    # we only optimize the parameters in the stitch
    for p in model1.parameters():
        p.requires_grad = False
    for p in model2.parameters():
        p.requires_grad = False

    # NOTE we may want to use vanilla gradient descent in the future but this should be
    # fine since it's default.
    optimizer = optim.Adadelta(stitch.parameters(), lr=DEFAULT_LR)
    scheduler = StepLR(optimizer, step_size=1, gamma=DEFAULT_LR_EXP_DROP)

    for _ in range(1, epochs):
        train1epoch(stitched_model, device, train_loader, optimizer)
        test(stitched_model, device, test_loader)
        scheduler.step()
        # TODO logging (and VISUALIZE here using tensorboardX... use a better format, once more,
        # than we did in "main")
    train1epoch(stitched_model, device, train_loader, optimizer)
    _, test_acc = test(stitched_model, device, test_loader)
    return test_acc

if __name__ == "__main__":
    train_loader, test_loader, device = init()

    # For each type of model, train DEFAULT_SANITY_INSTANCES of them (right under the import statement)
    for net_layers, net_valid_stitches, net_name in [
        (CT, CT2T, "CT"),
        # (C32, C32T32, "C32"),
    ]:
        print("Sanity checking stitches {}->{}".format(net_name, net_name))
        # models will store each model
        # stitches will store dictionaries of each layer to layer stitch (unique per pair: pivot, models[i])
        # penalties will store the difference in new loss minus minimum original loss
        # accs will store the accuracies of each of the trained models, originally
        # pivot_idx is the index of the models which is to be the pivot
        models = [None] * DEFAULT_SANITY_INSTANCES
        stitches = [None] * DEFAULT_SANITY_INSTANCES
        accs = [None] * DEFAULT_SANITY_INSTANCES
        penalties = [None] * DEFAULT_SANITY_INSTANCES
        pivot_idx = 0
        print("Training {} instances".format(DEFAULT_SANITY_INSTANCES))
        for i in range(DEFAULT_SANITY_INSTANCES):
            print("On Instance {}".format(i))
            models[i] =  Net(layers=net_layers).to(device)
            accs[i] = train_og(models[i], device, test_loader, train_loader, epochs=DEFAULT_EPOCHS_OG)
            stitches[i] = get_stitches(models[pivot_idx], models[i], net_valid_stitches)
            
            # Stitch each pair (pivot, model) and get the penalties
            penalties[i] = {}
            for l1_recv, l2_recvs in net_valid_stitches.items():
                penalties[i][l1_recv] = {}
                for l2_recv in l2_recvs:
                    l1_send = l1_recv - 1
                    l1_send = l2_recv - 1
                    assert l1_send > 0
                    assert l1_send > 0
                    print("Training a stitch from {} to {}".format(l1_recv-1, l2_recv-1))

                    st = stitches[i][l1_recv][l2_recv]
                    st_acc = train_stitch(models[pivot_idx], models[i], l1_recv, l2_recv, st, device, train_loader, test_loader, DEFAULT_EPOCHS_STITCH)
                    pen = min(accs[pivot_idx], accs[i]) - st_acc
                    penalties[i][l1_recv][l2_recv] = pen
                    print("Got acc {} and penalty {}".format(st_acc, pen))
        
        # NOTE we do sanity checks without tensors because of the asymmetry present
        # in certain cases like C32.

        # Initialize errors
        errors = [1 - acc for acc in accs]

        # Initialize avg penalties
        avg_penalties = {}
        for key, value in penalties[pivot_idx].items():
            avg_penalties[key] = {}
            for vkey in value.keys():
                avg_penalties[key][vkey] = 0.0
        # Fill avg penalties (sum) and then divide
        for key, value in avg_penalties.items():
            for vkey in avg_penalties.keys():
                for i in range(DEFAULT_SANITY_INSTANCES):
                    avg_penalties[key][vkey] += penalties[i][key][vkey]
                avg_penalties[key][vkey] /= DEFAULT_SANITY_INSTANCES
        
        # We are going to create a matrix which is indexxed by [i][j] where
        # i is the model (up to DEFAULT_SANITY_INSTANCES) and j is the layer in the model
        # that was self-stitched. We use two dictionaries: key2idx, and idx2key to remember
        # which idx in the matrix (in j: the row) corresponds to what layer
        key2idx = {}
        idx2key = {}
        matrix = [None] * DEFAULT_SANITY_INSTANCES
        for key, value in penalties[pivot_idx].items():
            if key in value:
                key2idx[key] = len(key2idx)
                idx2key[len(idx2key)] = key
        num_layers_self_stitch = len(key2idx)
        # Here we fill in the matrix
        for i in range(DEFAULT_SANITY_INSTANCES):
            matrix[i] = [penalties[i][idx2key[j]][idx2key[j]] for j in range(num_layers_self_stitch)]

        matrix = torch.tensor(matrix).float()
        avg_penalties_self = matrix.mean(0)
        l2_penalties_self = matrix.var(0)

        # The error percentage (1 - acc[i]) should be under 10%
        org = 0.1

        # The average penalty percentage (1 - penalty[:][l1][l2 = l1]) should be under 0.2 and the
        # specific penalty percentage (1 - penalty[i][l1][l2 = l1]) should be under 0.15
        epsilon = 0.2
        epsilon0 = 0.15

        # The average displacement and average squared displacement (variance) of 
        # (1 - penalty[:][l1][l1]) should be under 20% and 30%
        epsilon_L1 = 0.2
        epsilon_L2 = 0.3

        # The average penalty percentage (1 - penalty[:][l1][l2 != l1]) should be over 50%
        # (note that totally random is 10% accuracy which is a 90% error percentage)
        delta = 0.5
        
        # At this point all the penalties should be loaded in memory as should the accs, the models, and the stitches
        for i in range(DEFAULT_SANITY_INSTANCES):
            print("Model {} had error {} should < {}".format(i, errors[i], org))
        # Check that the exact self-stitching penalty is low
        for i in range(DEFAULT_SANITY_INSTANCES):
            for key, value in penalties[i].items():
                if key in value:
                    print("Model {} Penalty from layer {} to itself was {} should < {}".format(i, key, penalties[i][key][key], epsilon0))
        # Check that average self-stitching penalty is low
        for j in range(num_layers_self_stitch):
            print("Avg Penalty from layer {} to itself was {} should < {}".format(idx2key[j], avg_penalties_self[j], epsilon))
        # Check that the non-self stitching penalty is high
        for key, value in avg_penalties.items():
            for vkey, val in value.items():
                if key != vkey:
                    print("Avg Penalty from layer {} to {} was {} should > {}".format(key, vkey, avg_penalties[key][vkey], delta))
        
        
        # TODO L1
        # TODO visualizations
        # 1. Grid heatmap (of average penalties)
        # 2. Depth dimension of that grid ^ (i.e. not the average, but the distribution): one per i, j pair
        # TODO save the models (as state dicts or something else which does not depend on this namespace)
        # 1. models
        # 2. stitches