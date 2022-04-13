
import os
import time
import argparse
from pprint import PrettyPrinter
# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.cuda.amp import GradScaler, autocast

# Utility from our various different utility modules
from utils import (
    fix_seed,
    evaluate,
    adjust_learning_rate,
    combos
)
from download import (
    cifar_models_from_imagenet_models
)

from create_array_meta import (
    SMALLPAIRNUM2FILENAMES
)
from stitching_util import (
    Stitched,
    resnet18_34_layer2layer,
    resnet_small_small_layer2layer,
)
from resnet import (
    RESNETS_FOLDER,
    resnet18,
    resnet34,
    _resnet,
    BasicBlock
)

from img import (
    get_loaders,
    mean2_model_diff,
    label_before,
)

# Miscellaneous utility
pp = PrettyPrinter(indent=4)


def train_loop(model, train_loader, test_loader, parameters=None, epochs=None):
    # None signifies do all parameters (we might finetune single layers an that will speed up training)
    if parameters is None:
        parameters = list(model.parameters())

    optimizer = torch.optim.SGD(
        params=parameters,
        momentum=0.9,
        lr=args.lr * args.bsz / 256,
        weight_decay=args.wd
    )

    scaler = GradScaler()

    start = time.time()
    epochs = args.epochs if epochs is None else epochs
    for e in range(1, epochs + 1):
        model.train()
        # epoch
        # NOTE that enumerate's start changes the starting index
        for it, (inputs, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):

            # adjust
            adjust_learning_rate(epochs=epochs,
                                 warmup_epochs=args.warmup,
                                 base_lr=args.lr * args.bsz / 256,
                                 optimizer=optimizer,
                                 loader=train_loader,
                                 step=it)
            # zero grad
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                h = inputs
                h = model(h)
                loss = F.cross_entropy(h, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f'\t\tepoch: {e} | time: {time.time() - start:.3f}')

    eval_acc = evaluate(model, test_loader)

    return eval_acc, model

# We can shoot to do stitching with
# resnet18 (basic block)
# resnet34 (basic block)

# NOTE these are the experiments we have planned
# - resnet18 block2block (including within block sets: mistake I did before)
# - random resnet18 block2block with resnet18
# - resnet18 block2block w/ resnet34
# - random resnet18 block2block w/ resnet34

# NOTE right now this just finetunes the pretrained models so we can run our experiments tomorrow


def main_pretrain(args, res=True):
    if not res:
        raise NotImplementedError
    fix_seed(args.seed)

    train_loader, test_loader = get_loaders(args)
    models = cifar_models_from_imagenet_models()

    # Decide epochs such that they will make our model training slightly better
    desired_keys = ["resnet18", "resnet34", "resnet50", "wide_resnet50_2"]
    desired_epochs = [40, 60, 80, 100]
    desired_epochs = {desired_keys[i]: desired_epochs[i]
                      for i in range(len(desired_epochs))}

    # Remove undesired keys
    models = {key: models[key] for key in desired_keys}

    print(f"Only looking at models {list(models.keys())}")
    for model_name, model_and_meta in models.items():
        model, trained = model_and_meta
        if not trained:
            print(f"will train {model_name}")
            model = model.cuda()

            epochs = desired_epochs[model_name] if model_name in desired_epochs else None
            train_loop(model, train_loader, test_loader,
                       parameters=None, epochs=epochs)
            acc_percent = evaluate(model, test_loader)
            assert acc_percent >= 90, f"acc_percent was {acc_percent}"

            fname = os.path.join(RESNETS_FOLDER, f"{model_name}.pt")
            torch.save(model.state_dict(), fname)
    pass


# NOTE the use of lazy evaluation: this helps us test where we have less memory
# because it enables the garbage collector to garbage collect
def main_stitchtrain(args):
    # NOTE there are a total of 16 experiments for (r34, r18)
    # since you can pick first one r18 or r34
    # then yuo can pick the second one r18 or r34
    # then you can pick the first rand or not
    # then you can pick the second one rand or not
    ########################################################################################################################
    print("Generating resnet18 to resnet18 (and with random) stitches")

    def resnet18_resnet18_func(): return resnet18_34_layer2layer(
        sender18=True, reciever18=True)

    def resnet18_rand_resnet18_func(): return resnet18_34_layer2layer(
        sender18=True, reciever18=True)

    def resnet18_resnet18_rand_func(): return resnet18_34_layer2layer(
        sender18=True, reciever18=True)

    def resnet18_rand_resnet18_rand_func(): return resnet18_34_layer2layer(
        sender18=True, reciever18=True)

    print("Generating resnet18 to resnet34 (and with random) stitches")

    def resnet18_resnet34_func(): return resnet18_34_layer2layer(
        sender18=True, reciever18=False)

    def resnet18_rand_resnet34_func(): return resnet18_34_layer2layer(
        sender18=True, reciever18=False)

    def resnet18_resnet34_rand_func(): return resnet18_34_layer2layer(
        sender18=True, reciever18=False)

    def resnet18_rand_resnet34_rand_func(): return resnet18_34_layer2layer(
        sender18=True, reciever18=False)

    print("Generating resnet34 to resnet34 (and with random) stitches")

    def resnet34_resnet34_func(): return resnet18_34_layer2layer(
        sender18=False, reciever18=False)

    def resnet34_resnet34_rand_func(): return resnet18_34_layer2layer(
        sender18=False, reciever18=False)

    def resnet34_rand_resnet34_func(): return resnet18_34_layer2layer(
        sender18=False, reciever18=False)

    def resnet34_rand_resnet34_rand_func(): return resnet18_34_layer2layer(
        sender18=False, reciever18=False)

    print("Generating resnet34 -> resnet18 with rand")

    def resnet34_resnet18_func(): return resnet18_34_layer2layer(
        sender18=False, reciever18=True)

    def resnet34_resnet18_rand_func(): return resnet18_34_layer2layer(
        sender18=False, reciever18=True)

    def resnet34_rand_resnet18_func(): return resnet18_34_layer2layer(
        sender18=False, reciever18=True)
    def resnet34_rand_resnet18_rand_func(): return resnet18_34_layer2layer(
        sender18=False, reciever18=True)
########################################################################################################################
    N = 10  # len(resnet18_resnet18)
    print(f"Sims tables for resnet18 to resnet18 will be {N}x{N}")
    resnet18_resnet18_sims = [[0.0 for _ in range(N)] for _ in range(N)]
    resnet18_rand_resnet18_sims = [[0.0 for _ in range(N)] for _ in range(N)]
    resnet18_resnet18_rand_sims = [[0.0 for _ in range(N)] for _ in range(N)]
    resnet18_rand_resnet18_rand_sims = [
        [0.0 for _ in range(N)] for _ in range(N)]

    N = 10  # len(resnet18_resnet34)
    M = 18  # len(resnet18_resnet34[0])
    print(f"Sims tables for resnet18 to resnet34 will be {N}x{M}")
    resnet18_resnet34_sims = [[0.0 for _ in range(M)] for _ in range(N)]
    resnet18_rand_resnet34_sims = [[0.0 for _ in range(M)] for _ in range(N)]
    resnet18_resnet34_rand_sims = [[0.0 for _ in range(M)] for _ in range(N)]
    resnet18_rand_resnet34_rand_sims = [
        [0.0 for _ in range(M)] for _ in range(N)]

    N = 18  # len(resnet34_resnet34)
    print(f"Sims tables for resnet34 to resnet34 will be {N}x{N}")
    resnet34_resnet34_sims = [[0.0 for _ in range(N)] for _ in range(N)]
    resnet34_resnet34_rand_sims = [[0.0 for _ in range(N)] for _ in range(N)]
    resnet34_rand_resnet34_sims = [[0.0 for _ in range(N)] for _ in range(N)]
    resnet34_rand_resnet34_rand_sims = [
        [0.0 for _ in range(N)] for _ in range(N)]

    N = 18  # len(resnet34_resnet18)
    M = 10  # len(resnet34_resnet18[0])
    print(f"Sims tables for resnet34 to resnet18 will be {N}x{M}")
    resnet34_resnet18_sims = [[0.0 for _ in range(M)] for _ in range(N)]
    resnet34_resnet18_rand_sims = [[0.0 for _ in range(M)] for _ in range(N)]
    resnet34_rand_resnet18_sims = [[0.0 for _ in range(M)] for _ in range(N)]
    resnet34_rand_resnet18_rand_sims = [
        [0.0 for _ in range(M)] for _ in range(N)]
########################################################################################################################
    print("Loading resnets into memory from disk")

    r18 = resnet18(num_classes=10).cuda()
    # , map_location=torch.device('cpu')))
    r18.load_state_dict(torch.load(
        os.path.join(RESNETS_FOLDER, "resnet18.pt")))
    r18_rand = resnet18(num_classes=10).cuda()
    r34 = resnet34(num_classes=10).cuda()
    # , map_location=torch.device('cpu')))
    r34.load_state_dict(torch.load(
        os.path.join(RESNETS_FOLDER, "resnet34.pt")))
    r34_rand = resnet34(num_classes=10).cuda()

    print("Getting loaders")
    train_loader, test_loader = get_loaders(args)

    print("Confirming that acc is high")
    accp = 0.0
    accp = evaluate(r18, test_loader)
    r18_acc = accp / 100.0
    print(f"Accuracy of resnet18 is {accp}%")
    assert accp > 90

    accp = evaluate(r18_rand, test_loader)
    r18_rand_acc = accp / 100.0
    print(f"Accuracy of resnet18 random is {accp}%")
    assert accp < 20

    accp = evaluate(r34, test_loader)
    r34_acc = accp / 100.0
    print(f"Accuracy of resnet34 is {accp}%")
    assert accp > 90

    accp = evaluate(r34_rand, test_loader)
    r34_rand_acc = accp / 100.0
    print(f"Accuracy of resnet34 random is {accp}%")
    assert accp < 20
########################################################################################################################
    print("Generating similarity tables")

    folder = f"sims{args.fnum}"
    if not os.path.exists(folder):
        os.mkdir(folder)

    for sender, reciever, transformations_func, table, orig_acc1, orig_acc2, name in [
        # Resnet18 -> Resnet18
        (r18, r18, resnet18_resnet18_func, resnet18_resnet18_sims,
         r18_acc, r18_acc, "resnet18_resnet18"),
        (r18_rand, r18, resnet18_rand_resnet18_func, resnet18_rand_resnet18_sims,
         r18_acc, r18_rand_acc, "resnet18_rand_resnet18"),
        (r18, r18_rand, resnet18_resnet18_rand_func, resnet18_resnet18_rand_sims,
         r18_acc, r18_rand_acc, "resnet18_resnet18_rand"),
        (r18_rand, r18_rand, resnet18_rand_resnet18_rand_func, resnet18_rand_resnet18_rand_sims,
         r18_rand_acc, r18_rand_acc, "resnet18_rand_resnet18_rand"),

        # Resnet18 -> Resnet34
        (r18, r34, resnet18_resnet34_func, resnet18_resnet34_sims,
         r18_acc, r34_acc, "resnet18_resnet34"),
        (r18_rand, r34, resnet18_rand_resnet34_func, resnet18_rand_resnet34_sims,
         r18_rand_acc, r34_acc, "resnet18_rand_resnet34"),
        (r18_rand, r34_rand, resnet18_rand_resnet34_rand_func, resnet18_rand_resnet34_rand_sims,
         r18_rand_acc, r34_rand_acc, "resnet18_rand_resnet34_rand"),
        (r18, r34_rand, resnet18_resnet34_rand_func, resnet18_resnet34_rand_sims,
         r18_acc, r34_rand_acc, "resnet18_resnet34_rand"),

        # Resnet34 -> Resnet18
        (r34, r18, resnet34_resnet18_func, resnet34_resnet18_sims,
         r34_acc, r18_acc, "resnet34_resnet18"),
        (r34_rand, r18, resnet34_rand_resnet18_func, resnet34_rand_resnet18_sims,
         r34_rand_acc, r18_acc, "resnet34_rand_resnet18"),
        (r34, r18_rand, resnet34_resnet18_rand_func, resnet34_resnet18_rand_sims,
         r34_acc, r18_rand_acc, "resnet34_resnet18_rand"),
        (r34_rand, r18_rand, resnet34_rand_resnet18_rand_func, resnet34_rand_resnet18_rand_sims,
         r34_rand_acc, r18_rand_acc, "resnet34_rand_resnet18_rand"),

        # Resnet34 -> Resnet34
        (r34, r34, resnet34_resnet34_func, resnet34_resnet34_sims,
         r34_acc, r34_acc, "resnet34_resnet34"),
        (r34_rand, r34, resnet34_rand_resnet34_func, resnet34_rand_resnet34_sims,
         r34_rand_acc, r34_acc, "resnet34_rand_resnet34"),
        (r34, r34_rand, resnet34_resnet34_rand_func, resnet34_resnet34_rand_sims,
         r34_acc, r34_rand_acc, "resnet34_resnet34_rand"),
        (r34_rand, r34_rand, resnet34_rand_resnet34_rand_func, resnet34_rand_resnet34_rand_sims,
         r34_rand_acc, r34_rand_acc, "resnet34_rand_resnet34_rand"),
    ]:

        N = len(table)
        M = len(table[0])
        transformations, _, idx2label = transformations_func()

        # NOTE we ignore first and last layer (for reciever and send)
        # NOTE outfrom, into format
        for i in range(N - 1):
            for j in range(1, M):
                try:
                    acc = 0.0
                    orig_acc = min(orig_acc1, orig_acc2)

                    snd_label, rcv_label = idx2label[(i, j)]
                    print(
                        f"\tstitching {i}->{j} which is {snd_label}->{rcv_label}")
                    st = transformations[i][j].cuda()
                    model = Stitched(sender, reciever,
                                     snd_label, rcv_label, st)
                    acc, _ = train_loop(
                        model, train_loader, test_loader, epochs=4, parameters=list(st.parameters()))
                    acc /= 100.0
                    print(f"\tAccuracy of model is {acc}")
                    print(f"\tOriginal accuracy was {orig_acc}")
                    table[i][j] = acc

                    filename_stitch = name + f"{i}_{j}_stitch.pt"
                    filename_stitch = os.path.join(folder, filename_stitch)
                    print(
                        f"\tSaving stitch {i}->{j} to filename {filename_stitch}")
                    torch.save(transformations[i][j].cpu(
                    ).state_dict(), filename_stitch)
                except:
                    table[i][j] = -1

        filename_sims = name + "_sims.pt"
        filename_sims = os.path.join(folder, filename_sims)
        print(f"Saving sims to {filename_sims}")
        torch.save(torch.tensor(table), filename_sims)
########################################################################################################################
    pass

# Train a bunch of small resnets: all combinations from 1, 1, 1, 1 to 2, 2, 2, 2
# (there are 16 combinations)


def main_train_small(args, res=True):
    if not res:
        raise NotImplementedError
    train_loader, test_loader = get_loaders(args)

    combinations = combos(4, [1, 2])
    for combination in combinations:
        x, y, z, w = combination
        fname = os.path.join(RESNETS_FOLDER, f"resnet_{x}{y}{z}{w}.pt")
        if os.path.exists(fname):
            print(f"Skipping {fname}")
            continue
        else:

            # NOT pretrained and NO progress bar (these aren't supported anyways)
            model = _resnet(
                f"resnet_{x}{y}{z}{w}", BasicBlock, combination, False, False, num_classes=10)
            model = model.cuda()

            print(f"will train net {x}{y}{z}{w}")

            # TODO play around with epochs
            epochs = 40
            train_loop(model, train_loader, test_loader,
                       parameters=None, epochs=epochs)
            acc_percent = evaluate(model, test_loader)
            print(f"acc_percent was {acc_percent}")

            # NOTE this is ../pretrained_resnet
            torch.save(model.state_dict(), fname)
    pass


# TODO this should not be to difficult to modularize?
def train_sim_loss(stitch, sender, reciever, snd_label, rcv_label, model_blocks, train_loader, test_loader, epochs=3):
    # None signifies do all parameters (we might finetune single layers an that will speed up training)
    parameters = list(stitch.parameters())

    label_before_rcv_label = label_before(rcv_label, model_blocks)

    # NOTE this is copied from the train_loop method
    optimizer = torch.optim.SGD(
        params=parameters,
        momentum=0.9,
        lr=args.lr * args.bsz / 256,
        weight_decay=args.wd
    )

    # NOTE this is copied
    scaler = GradScaler()
    start = time.time()
    for e in range(1, epochs + 1):
        stitch.train()
        for it, (inputs, y) in enumerate(train_loader, start=(e - 1) * len(train_loader)):
            # NOTE this is copied
            adjust_learning_rate(epochs=epochs,
                                 warmup_epochs=args.warmup,
                                 base_lr=args.lr * args.bsz / 256,
                                 optimizer=optimizer,
                                 loader=train_loader,
                                 step=it)
            # zero grad
            optimizer.zero_grad(set_to_none=True)

            # NOTE this is NOT copied and is instead the precise thing we change
            # It is NOT optimized (ideally you should cache, and we will do that
            # once we have more time to clean up the code)
            ###############################
            with autocast():
                # Run both the sender and reciever up to the send label
                stitch_input = sender(inputs, vent=snd_label, into=False)
                stitch_output = stitch(stitch_input)
                stitch_target = reciever(
                    inputs, vent=label_before_rcv_label, apply_post=True, into=False)
                loss = F.mse_loss(stitch_output, stitch_target)
            ###############################

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f'\t\tepoch: {e} | time: {time.time() - start:.3f}')

    # Make a model to get the loss on the actual task
    model = Stitched(sender, reciever, snd_label, rcv_label, stitch)
    eval_acc = evaluate(model, test_loader)
    return eval_acc, model


def main_stitchtrain_small(args):
    file1, file2 = SMALLPAIRNUM2FILENAMES[int(args.smallpairnum)]
    numbers1 = list(map(int, file1.split(".")[0][-4:]))
    numbers2 = list(map(int, file2.split(".")[0][-4:]))
    name1 = "resnet_" + "".join(map(str, numbers1))
    name2 = "resnet_" + "".join(map(str, numbers2))
    output_folder = f"sims_{name1}_{name2}"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print(f"Will stitch {file1} and {file2} in {RESNETS_FOLDER}")
    print(f"numbers1: {numbers1}")
    print(f"numbers2: {numbers2}")
    print(f"name1: {name1}")
    print(f"name2: {name2}")

    file1 = os.path.join(RESNETS_FOLDER, file1)
    file2 = os.path.join(RESNETS_FOLDER, file2)

    print("Loading models")
    model1 = _resnet(name1, BasicBlock, numbers1, False, False, num_classes=10)
    model2 = _resnet(name2, BasicBlock, numbers2, False, False, num_classes=10)
    model1_rand = _resnet(name1 + "_rand", BasicBlock,
                          numbers1, False, False, num_classes=10)
    model2_rand = _resnet(name2 + "_rand", BasicBlock,
                          numbers2, False, False, num_classes=10)
    model1.cuda()
    model2.cuda()
    model1_rand.cuda()
    model2_rand.cuda()
    model1.load_state_dict(torch.load(file1))
    model2.load_state_dict(torch.load(file2))

    print("Getting loaders")
    train_loader, test_loader = get_loaders(args)

    print("Evaluating accuracies of pretrained models")
    accp = 0.0
    accp = evaluate(model1, test_loader)
    model1_acc = accp / 100.0
    print(f"Accuracy of {name1} is {accp}%")
    assert accp > 90

    # NOTE: this often comes out to zero and I don't understand why
    # shouldn't it in theory be around 10% ?
    accp = evaluate(model1_rand, test_loader)
    model1_rand_acc = accp / 100.0
    print(f"Accuracy of {name1} random is {accp}%")
    assert accp < 20

    accp = evaluate(model2, test_loader)
    model2_acc = accp / 100.0
    print(f"Accuracy of {name2} is {accp}%")
    assert accp > 90

    # NOTE: this often comes out to zero and I don't understand why
    # shouldn't it in theory be around 10% ?
    accp = evaluate(model2_rand, test_loader)
    model2_rand_acc = accp / 100.0
    print(f"Accuracy of {name2} random is {accp}%")
    assert accp < 20

    # Number of blocks/layers to stitch in model 1 and model 2
    N1 = sum(numbers1) + 2
    N2 = sum(numbers2) + 2

    model1_model2_sims = [[0.0 for _ in range(N2)] for _ in range(N1)]
    model1_rand_model2_sims = [[0.0 for _ in range(N2)] for _ in range(N1)]
    model1_model2_rand_sims = [[0.0 for _ in range(N2)] for _ in range(N1)]
    model1_rand_model2_rand_sims = [
        [0.0 for _ in range(N2)] for _ in range(N1)]

    # NOTE these store the mean squared error of the vanilla stitch with the expected representation
    vanilla_rep_model1_model2_mean2 = [
        [0.0 for _ in range(N2)] for _ in range(N1)]
    vanilla_rep_model1_rand_model2_mean2 = [
        [0.0 for _ in range(N2)] for _ in range(N1)]
    vanilla_rep_model1_model2_rand_mean2 = [
        [0.0 for _ in range(N2)] for _ in range(N1)]
    vanilla_rep_model1_rand_model2_rand_mean2 = [
        [0.0 for _ in range(N2)] for _ in range(N1)]

    print("Initializing the stitches (note all idx2label are the same)")
    model1_model2_stitches, _, idx2label = resnet_small_small_layer2layer(
        numbers1, numbers2)
    model1_rand_model2_stitches, _, _ = resnet_small_small_layer2layer(
        numbers1, numbers2)
    model1_model2_rand_stitches, _, _ = resnet_small_small_layer2layer(
        numbers1, numbers2)
    model1_rand_model2_rand_stitches, _, _ = resnet_small_small_layer2layer(
        numbers1, numbers2)

    # NOTE: This is an addition for the autoencoder test
    ################################################################################
    model1_model2_autoencoder_stitches, _, _ = resnet_small_small_layer2layer(
        numbers1, numbers2)
    model1_rand_model2_autoencoder_stitches, _, _ = resnet_small_small_layer2layer(
        numbers1, numbers2)
    model1_model2_rand_autoencoder_stitches, _, _ = resnet_small_small_layer2layer(
        numbers1, numbers2)
    model1_rand_model2_rand_autoencoder_stitches, _, _ = resnet_small_small_layer2layer(
        numbers1, numbers2)

    # NOTE these store the accuracies (of the stitched models) of the autoencoder stitches
    model1_model2_autoencoder_sims = [
        [0.0 for _ in range(N2)] for _ in range(N1)]
    model1_rand_model2_autoencoder_sims = [
        [0.0 for _ in range(N2)] for _ in range(N1)]
    model1_model2_rand_autoencoder_sims = [
        [0.0 for _ in range(N2)] for _ in range(N1)]
    model1_rand_model2_rand_autoencoder_sims = [
        [0.0 for _ in range(N2)] for _ in range(N1)]

    # NOTE these store the mean squared error of the autoencoder stitches
    autoencoder_rep_model1_model2_mean2 = [
        [0.0 for _ in range(N2)] for _ in range(N1)]
    autoencoder_rep_model1_rand_model2_mean2 = [
        [0.0 for _ in range(N2)] for _ in range(N1)]
    autoencoder_rep_model1_model2_rand_mean2 = [
        [0.0 for _ in range(N2)] for _ in range(N1)]
    autoencoder_rep_model1_rand_model2_rand_mean2 = [
        [0.0 for _ in range(N2)] for _ in range(N1)]

    # NOTE these store the mean squared error of the autoencoder stitches with the vanilla stitches
    vanilla_autoencoder_model1_model2_mean2 = [
        [0.0 for _ in range(N2)] for _ in range(N1)]
    vanilla_autoencoder_model1_rand_model2_mean2 = [
        [0.0 for _ in range(N2)] for _ in range(N1)]
    vanilla_autoencoder_model1_model2_rand_mean2 = [
        [0.0 for _ in range(N2)] for _ in range(N1)]
    vanilla_autoencoder_model1_rand_model2_rand_mean2 = [
        [0.0 for _ in range(N2)] for _ in range(N1)]
    ################################################################################

    pp.pprint(idx2label)

    print(f"Stitching, will save in {output_folder}")
    for sender, reciever,\
            stitches, stitches_autoencoder,\
            sims, autoencoder_sims,\
            vanilla_rep_mean2, autoencoder_rep_mean2, vanilla_autoencoder_mean2,\
            orig_acc1, orig_acc2, name in [
                # Resnet18 -> Resnet18
                (model1, model2,
                 model1_model2_stitches, model1_model2_autoencoder_stitches,
                 model1_model2_sims, model1_model2_autoencoder_sims,
                 vanilla_rep_model1_model2_mean2, autoencoder_rep_model1_model2_mean2, vanilla_autoencoder_model1_model2_mean2,
                 model1_acc, model2_acc, f"{name1}_{name2}"),
                # Resnet18 random -> Resnet18
                (model1_rand, model2,
                 model1_rand_model2_stitches, model1_rand_model2_autoencoder_stitches,
                 model1_rand_model2_sims, model1_rand_model2_autoencoder_sims,
                 vanilla_rep_model1_rand_model2_mean2, autoencoder_rep_model1_rand_model2_mean2, vanilla_autoencoder_model1_rand_model2_mean2,
                 model1_rand_acc, model2_acc, f"{name1}_rand_{name2}"),
                # Resnet18 -> Resnet18 random
                (model1, model2_rand,
                 model1_model2_rand_stitches, model1_model2_rand_autoencoder_stitches,
                 model1_model2_rand_sims, model1_model2_rand_autoencoder_sims,
                 vanilla_rep_model1_model2_rand_mean2, autoencoder_rep_model1_model2_rand_mean2, vanilla_autoencoder_model1_model2_rand_mean2,
                 model1_acc, model2_rand_acc, f"{name1}_{name2}_rand"),
                # Resnet18 random -> Resnet18 random
                (model1_rand, model2_rand,
                 model1_rand_model2_rand_stitches, model1_rand_model2_rand_autoencoder_stitches,
                 model1_rand_model2_rand_sims, model1_rand_model2_rand_autoencoder_sims,
                 vanilla_rep_model1_rand_model2_rand_mean2, autoencoder_rep_model1_rand_model2_rand_mean2, vanilla_autoencoder_model1_rand_model2_rand_mean2,
                 model1_rand_acc, model2_rand_acc, f"{name1}_rand_{name2}_rand")
            ]:
        print(f"Stitching {name}")
        # NOTE we ignore first and last layer (for reciever and send)
        # NOTE outfrom, into format
        for i in range(N1 - 1):
            for j in range(N2):
                try:
                    acc = 0.0

                    snd_label, rcv_label = idx2label[(i, j)]
                    print(
                        f"\tstitching {i}->{j} which is {snd_label}->{rcv_label}")
                    print(
                        f"\tOriginal accuracies were {orig_acc1} and {orig_acc2}")

                    vanilla_stitch = stitches[i][j].cuda()
                    vanilla_model = Stitched(
                        sender, reciever, snd_label, rcv_label, vanilla_stitch)

                    # Epochs can be changed (large may lead to exploding gradients?)
                    vanilla_acc, _ = train_loop(
                        vanilla_model, train_loader, test_loader, epochs=4, parameters=list(vanilla_stitch.parameters()))
                    vanilla_acc /= 100.0
                    print(
                        f"\tAccuracy of vanilla stitch model is {vanilla_acc}")

                    sims[i][j] = acc

                    # Vanilla stitch output difference the stitches learned at layers [i][j]
                    # NOTE that model blocks is numbers2 because we feed INTO model2
                    vanilla_rep_mean2_diff = mean2_model_diff(
                        vanilla_stitch, sender, reciever, snd_label, rcv_label, numbers2, train_loader, stitch2=None)
                    vanilla_rep_mean2[i][j] = vanilla_rep_mean2_diff
                    print(
                        f"\tVanilla stitch mean2 difference is {vanilla_rep_mean2_diff}")

                    autoencoder_stitch = stitches_autoencoder[i][j].cuda()
                    # TODO train it on the similarity loss of the expected representation
                    # train_sim_loss(autoencoder_stitch, sender, reciever, snd_label, rcv_label, numbers2, train_loader, test_loader, epochs=30)
                    autoencoder_acc, _ = 0.0, None
                    print(
                        f"\tAccuracy of autoencoder stitch model is {autoencoder_acc}")
                    autoencoder_sims[i][j] = acc

                    autoencoder_rep_mean2_diff = mean2_model_diff(
                        autoencoder_stitch, sender, reciever, snd_label, rcv_label, numbers2, train_loader)
                    autoencoder_rep_mean2[i][j] = autoencoder_rep_mean2_diff
                    print(
                        f"\tAutoencoder stitch mean2 difference is {autoencoder_rep_mean2_diff}")

                    vanilla_autoencoder_mean2_diff = mean2_model_diff(
                        vanilla_stitch, sender, reciever, snd_label, rcv_label, numbers2, train_loader, stitch2=autoencoder_stitch)
                    vanilla_autoencoder_mean2[i][j] = vanilla_autoencoder_mean2_diff
                    print(
                        f"\tVanilla stitch autoencoder mean2 difference is {vanilla_autoencoder_mean2_diff}")
                    print("***")

                    if j == 0:
                        foldername_images = os.path.join(
                            output_folder, f"images_{i}_{j}")
                        print(f"Saving 5 random images to {foldername_images}")
                        if not os.path.exists(foldername_images):
                            os.mkdir(foldername_images)
                        # TODO
                        # save_random_image_pairs(stitches[i][j].cuda(
                        # ), sender, snd_label, 5, foldername_images, train_loader)
                    pass

                except Exception as e:
                    print(f"Failed on layers {i}, {j} ({e})")
                    sims[i][j] = 0.0
                    raise e
                # these are tests
                break
                pass
            # these are tests
            break
            pass

        # Accuracies of stitched models generated
        filename_sims = name + "_sims.pt"
        filename_autoencoder_sims = name + "_autoencoder_sims.pt"

        # Differences between what was learned and what was expected (roughly)
        filename_vanilla_autoencoder_mean2 = name + "_vanilla_autoencoder_mean2.pt"
        filename_autoencoder_rep_mean2 = name + "_autoencoder_rep_mean2.pt"
        filename_vanilla_rep_mean2 = name + "_vanilla_rep_mean2.pt"

        filename_sims = os.path.join(output_folder, filename_sims)
        filename_autoencoder_sims = os.path.join(
            output_folder, filename_autoencoder_sims)
        filename_vanilla_autoencoder_mean2 = os.path.join(
            output_folder, filename_vanilla_autoencoder_mean2)
        filename_autoencoder_rep_mean2 = os.path.join(
            output_folder, filename_autoencoder_rep_mean2)
        filename_vanilla_rep_mean2 = os.path.join(
            output_folder, filename_vanilla_rep_mean2)

        print(f"Saving sims to {filename_sims}")
        torch.save(torch.tensor(sims), filename_sims)
        print(f"Saving autoencoder sims to {filename_autoencoder_sims}")
        torch.save(torch.tensor(autoencoder_sims), filename_autoencoder_sims)
        print(
            f"Saving vanilla autoencoder mean2 to {filename_vanilla_autoencoder_mean2}")
        torch.save(torch.tensor(vanilla_autoencoder_mean2),
                   filename_vanilla_autoencoder_mean2)
        print(
            f"Saving autoencoder rep mean2 to {filename_autoencoder_rep_mean2}")
        torch.save(torch.tensor(autoencoder_rep_mean2),
                   filename_autoencoder_rep_mean2)
        print(f"Saving vanilla rep mean2 to {filename_vanilla_rep_mean2}")
        torch.save(torch.tensor(vanilla_rep_mean2), filename_vanilla_rep_mean2)

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--name', default=None, type=str)

    parser.add_argument('--bsz', default=256, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--warmup', default=10, type=int)
    # NOTE this is modified
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--wd', default=0.01, type=float)

    parser.add_argument('--encoder', default='random-resnet18', type=str, choices=['random-resnet18',
                                                                                   'esimclr-cifar10-resnet18',
                                                                                   'esimclr-imagenet-resnet50',
                                                                                   'esimclr-cifar100-resnet18'])
    parser.add_argument('--fraction', default=1.0, type=float)
    parser.add_argument('--dataset', default='cifar10',
                        type=str, choices=['cifar10', 'cifar100'])
    # Used by Resnet18 - Resnet34 experiments to do multiple copies with an id
    parser.add_argument('--fnum', default='0', type=str)
    # Used by Resnet 1111 -> 2222 to select which pair num to use
    parser.add_argument('--smallpairnum', default='1', type=str)
    # here you can also select whether to use resnet or not
    parser.add_argument('--train_mode', default='stitchtrain_small', type=str, choices=[
        'stitchtrain_small',
        'stitchtrain_r18_r34',
        'train_small',
        'train_small_no_res',
        'train_r18_r34',
        'train_r18_r34_no_res',
    ])
    args = parser.parse_args()

    # TODO
    # 1. Enable each of these to be run with either cifar100 or cifar10, and then make the batch
    #    tests run both dataasets (should store outputs in same folder)
    # 2. Enable resnets to not have residuals (just flip a flag in the parent class and then in
    #    the child class)
    # 3. Enable the training of resnets that have no residual (just append that to the function
    #    that trains the resnets, changing the filename from "resnet" to "noresnet")
    # 4. Enable the full experiments to load both resnets and noresnets by simplying adding those filenames
    #    to the massive dictionary in create_array_meta.py. NOTE that this will quadruple our number of tests
    #    from 256 to 1024. Moreover that is further doubled by the fact we try two datasets: 2048. If I were to
    #    stitch cifar100-trained resnets to cifar10-trained resnets this would further explode, so it's probably
    #    worth
    #     4.1 Optimizing the similarity loss tester to use a dataloader that uses cached representations
    #     4.2 Doing only a subset of the tests (that is random) or running them in a random order
    #     4.3 Running the full test-set during the course of my spring break.
    # 5. Updating the heatmap code to be able to generate all the heatmaps and then CLOSE the pyplot window to
    #    avoid running out of memory and getting those warnings.

    # Train stitches (rather generic)
    if args.train_mode == "stitchtrain_small":
        main_stitchtrain_small(args)
    elif args.train_mode == "stitchtrain_r18_r34":
        main_stitchtrain(args)
    # Pretrain (i.e. train the models we will freeze)
    elif args.train_mode == "train_small":
        main_train_small(args)
    elif args.train_mode == "train_r18_r34":
        main_pretrain(args)
    # Pretrain without residuals
    elif args.train_mode == "train_small_no_res":
        main_train_small(args, res=False)
    elif args.train_mode == "train_r18_r34_no_res":
        main_pretrain(args, res=False)
    else:
        raise NotImplementedError
