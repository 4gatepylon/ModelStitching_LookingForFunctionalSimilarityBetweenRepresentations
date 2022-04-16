
# Enables type annotations using enclosing classes
from __future__ import annotations
from img import (
    get_loaders,
    mean2_model_diff,
    label_before,
)
from resnet import (
    RESNETS_FOLDER,
    resnet18,
    resnet34,
    _resnet,
    BasicBlock
)
from stitching_util import (
    Stitched,
    resnet18_34_layer2layer,
    resnet_small_small_layer2layer,
)
from create_array_meta import (
    SMALLPAIRNUM2FILENAMES
)
from download import (
    cifar_models_from_imagenet_models
)
from utils import (
    fix_seed,
    evaluate,
    adjust_learning_rate,
    combos
)
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torch.nn.functional as F
from pprint import PrettyPrinter
import argparse
import time
import os

import unittest

# Enables more interesting type annotations
from typing_extensions import (
    Concatenate,
    ParamSpec,
)
from typing import (
    NoReturn,
    Callable,
    Union,
    List,
    Tuple,
    TypeVar,
)

import torch
import torch.nn as nn
import random
import numpy as np

from table import Table

T = TypeVar('T')
G = TypeVar('G')
P = ParamSpec('P')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def combos(length, possibles):
    if (length == 1):
        return [[x] for x in possibles]
    combinations = []
    for possible in possibles:
        remainders = combos(length - 1, possibles)
        for remainder in remainders:
            combinations.append(remainder + [possible])
    return combinations


def all_net_pairs():
    combinations = combos(4, [1, 2])
    assert len(combinations) == 16
    mapping = {}
    num = 1
    for i in range(16):
        for j in range(16):
            mapping[num] = (
                "resnet"+"".join(map(lambda c: str(c), combinations[i]))+".pt",
                "resnet"+"".join(map(lambda c: str(c), combinations[j]))+".pt"
            )
            num += 1
    return mapping, num


class PairExp(object):
    """ A part of an experiment that is for a pair of networks """

    def __init__(
            self: PairExp,
            layers1: List[int],
            layers2: List[int],
    ) -> NoReturn:
        # TODO
        # 1. load the networks
        # 2. load the random networks (controls)
        # 3. create the stitch table
        # 4. create the stitched network table
        # 5. (below) use those tables and the traininer to
        #    get the similarities
        self.send_recv_sims = Table.mappedTable(None, None)
        self.send_rand_recv_sims = Table.mappedTable(None, None)
        self.send_recv_rand_sims = Table.mappedTable(None, None)
        self.send_rand_recv_rand_sims = Table.mappedTable(None, None)


class Experiment(object):
    def __init__(self: Experiment) -> NoReturn:
        pass


""" Old utility that has to be changed! """


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
