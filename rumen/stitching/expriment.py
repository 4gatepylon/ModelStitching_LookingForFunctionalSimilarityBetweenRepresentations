
# Enables type annotations using enclosing classes
from __future__ import annotations

from torch.cuda.amp import autocast
import torch.nn.functional as F
import os

import unittest

# Enables more interesting type annotations
from typing_extensions import (
    ParamSpec,
)
from typing import (
    NoReturn,
    Any,
    Callable,
    Union,
    Dict,
    List,
    Tuple,
    TypeVar,
)

import torch
import torch.nn as nn
import random
import numpy as np

# Utility that I have written
from table import Table
from stitched_resnet import StitchedResnet
from torch.utils.data import DataLoader
from layer_label import LayerLabel
from resnet.resnet import Resnet, BasicBlock
from resnet.resnet_utils import Identity
from rep_shape import RepShape
from loaders import Loaders
from resnet.resnet_generator import ResnetGenerator
from trainer import Trainer, Hyperparams

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


def choose_product(possibles: List[T], length: int) -> List[List[T]]:
    """ All ordered subsequences of length `length` of where each element is in `possibles` """
    if (length == 1):
        return [[x] for x in possibles]
    combinations = []
    for possible in possibles:
        remainders = choose_product(length - 1, possibles)
        for remainder in remainders:
            combinations.append(remainder + [possible])
    return combinations


def choose_unordered_subset(items: List[T], k: int) -> List[T]:
    """ All unordered subsets (as lists) of `items`` of size `k`. The `choose` function is a helper. """
    def choose(items: List[T], k: int, index: int) -> List[T]:
        if len(items) - index < k:
            return []
        elif k == 1:
            return [[items[i]] for i in range(len(items) - index, len(items), 1)]
        else:
            not_choosing: List[List[T]] = choose(items, k, index=index + 1)
            choosing: List[List[T]] = choose(items, k - 1, index=index + 1)
            for choice in choosing:
                choice.append(items[index])
            return not_choosing + choosing
    return choose(items, k, 0)


def all_net_pairs():
    combinations = choose_product([1, 2], 4)
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

    # TODO change the naming and actually use/test

    @staticmethod
    def mean2_multiple_stitches(
        sender: Resnet,
        reciever: Resnet,
        stitch_tables: List[List[List[nn.Module]]],
        labels: List[List[Tuple[LayerLabel, LayerLabel]]],
        train_loader: DataLoader,
    ) -> Tuple[Dict[Tuple[int, int], List[List[float]]], List[List[List[float]]]]:
        """
        Given a list of stitch tables, return the mean2 difference between
        every pair of stitches pointwise for every pair of stitch tables (on the sender)
        as well as the list of mean2 differences between each table in the stitch_tables
        and the original network's expected representation (that of the reciever, output
        from the layer before that recieving the stitch). The dictionary keys are the
        indices of the labels list, while indices of the list of tables (comparing with
        original network) correspond to the indices in the stitch tables.

        Example Usage:
        mean2_multiple_stitches(
            sender,
            reciever,
            [vanilla_stitches, sim_stitches],
            labels,
            train_loader,
        )
        """

        # 1. compare 2 stitches
        #    - sender => sender, reciever => sender
        #    - send_stitch is the stitch, recv_stitch is the other stitch
        # 2. compare original with OG
        #    - sender => sender, reciever => reciever
        #    - send_stitch is the stitch
        #    - recv_stitch is an identity function

        # Choose all stitches and compare them to the original
        identity = Identity()
        identity_stitch_table: List[List[nn.Module]] = \
            [[identity for _ in range(len(stitch_tables[0]))]
             for _ in range(len(stitch_tables))]

        mean2_original: List[List[List[float]]] = [
            PairExp.mean2_model_model(
                sender,
                reciever,
                stitch_table,
                identity_stitch_table,
                labels,
                train_loader,
            )
            for stitch_table in stitch_tables
        ]

        # Choose all unordered pairs of stitches and compare them
        stitch_pairs: List[List[List[List[nn.Module]]]] = \
            choose_unordered_subset(stitch_tables, 2)
        stitch_indices: List[List[int]] = \
            choose_unordered_subset(list(range(len(stitch_tables))), 2)

        mean2_stitches: Dict[Tuple[int, int], List[List[float]]] = {}
        for stitch_pair, index_pair in zip(stitch_pairs, stitch_indices):
            stitch_table1, stitch_table2 = stitch_pair
            index1, index2 = index_pair
            mean2_table = PairExp.mean2_model_model(
                sender,
                sender,
                stitch_table1,
                stitch_table2,
                labels,
                train_loader,
            )
            mean2_stitches[(index1, index2)] = mean2_table
        return (mean2_stitches, mean2_original)

    @ staticmethod
    def mean2_model_model(
        sender: Resnet,
        reciever: Resnet,
        send_stitches: List[List[nn.Module]],
        recv_stitches: List[List[nn.Module]],
        labels: List[List[Tuple[LayerLabel, LayerLabel]]],
        train_loader: DataLoader
    ) -> List[List[float]]:
        assert len(labels) > 0
        assert len(labels) == len(send_stitches)
        assert len(labels) == len(recv_stitches)
        assert len(labels[0]) > 0
        assert len(labels[0]) == len(send_stitches[0])
        assert len(labels[0]) == len(recv_stitches[0])

        mean2_table: List[List[float]] = [
            [0.0 for _ in range(len(labels[0]))] for _ in range(len(labels))]

        for i in range(len(labels)):
            for j in range(len(labels[i])):
                send_label, recv_label = labels[i][j]
                send_stitch, recv_stitch = send_stitches[i][j], recv_stitches[i][j]
                mean2_table[i][j] = PairExp.mean2_model_diff(
                    send_stitch,
                    recv_stitch,
                    sender,
                    reciever,
                    send_label,
                    recv_label,
                    train_loader,
                )
        return mean2_table

    @staticmethod
    def mean2_layer_layer(
            send_stitch: nn.Module,
            recv_stitch: nn.Module,
            sender: Resnet,
            reciever: Resnet,
            send_label: LayerLabel,
            recv_label: LayerLabel,
            train_loader: DataLoader,
    ) -> float:
        num_images = len(train_loader)
        total = 0.0
        recv_label = recv_label - 1
        for x, _ in train_loader:
            with autocast():
                # TODO autocase and should we `pool_and_flatten``
                sent = sender.outfrom_forward(x, send_label)
                recv = reciever.outfrom_forward(x, recv_label)

                sent_stitch = send_stitch(sent)
                recv_stitch = recv_stitch(recv)

                diff = (sent_stitch - recv_stitch).pow(2).mean()
                total += diff.cpu().item()
            pass
        # Average over the number of images
        total /= num_images
        return total


class Experiment(object):
    RESNETS_FOLDER = "../../resnets/"
    SIMS_FOLDER = "../../sims/"

    def __init__(self: Experiment) -> NoReturn:
        pass

    @staticmethod
    def pretrain(args: Any) -> NoReturn:
        """
        Train all resnets of the form[x, y, z, w] where x is 1 or 2, y is 1 or 2, etcetera...
        unless they are already trained (in which case we will use them later). Store them in
        the resnets folder.
        """
        train_loader, test_loader = Loaders.get_loaders_ffcv(args)

        combinations = choose_product([1, 2], 4)
        for combination in combinations:
            x, y, z, w = combination
            fname = os.path.join(
                Experiment.RESNETS_FOLDER, f"resnet_{x}{y}{z}{w}.pt",
            )
            if os.path.exists(fname):
                print(f"Skipping {fname}")
                continue
            else:

                # NOT pretrained and NO progress bar (these aren't supported anyways)
                model = ResnetGenerator.generate(
                    f"resnet_{x}{y}{z}{w}", BasicBlock, combination, False, False, num_classes=10)
                model = model.cuda()

                print(f"will train net {x}{y}{z}{w}")

                epochs = 40
                Trainer.train_loop(
                    model,
                    train_loader,
                    test_loader,
                    parameters=None,
                    epochs=epochs,
                )
                acc_percent = Trainer.evaluate(
                    model,
                    test_loader
                )
                print(f"acc_percent was {acc_percent}")

                # NOTE this is ../../pretrained_resnet
                torch.save(model.state_dict(), fname)

    @staticmethod
    def stitchtrain(args: Any, filename_pair: Tuple[str, str]) -> NoReturn:
        # TODO refactor this
        file1, file2 = filename_pair
        numbers1 = list(map(int, file1.split(".")[0][-4:]))
        numbers2 = list(map(int, file2.split(".")[0][-4:]))
        name1 = "resnet_" + "".join(map(str, numbers1))
        name2 = "resnet_" + "".join(map(str, numbers2))
        output_folder = f"sims_{name1}_{name2}"
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        print(
            f"Will stitch {file1} and {file2} in {Experiment.RESNETS_FOLDER}")
        print(f"numbers1: {numbers1}")
        print(f"numbers2: {numbers2}")
        print(f"name1: {name1}")
        print(f"name2: {name2}")

        file1 = os.path.join(Experiment.RESNETS_FOLDER, file1)
        file2 = os.path.join(Experiment.RESNETS_FOLDER, file2)

        print("Loading models")
        model1 = ResnetGenerator.generate(
            name1,
            BasicBlock,
            numbers1,
            False,
            False,
            num_classes=10,
        )
        model2 = ResnetGenerator.generate(
            name2,
            BasicBlock,
            numbers2,
            False,
            False,
            num_classes=10,
        )
        model1_rand = ResnetGenerator.generate(
            name1 + "_rand",
            BasicBlock,
            numbers1,
            False,
            False,
            num_classes=10,
        )
        model2_rand = ResnetGenerator.generate(
            name2 + "_rand",
            BasicBlock,
            numbers2,
            False,
            False,
            num_classes=10,
        )

        print("Converting everything to Cuda")
        model1.cuda()
        model2.cuda()
        model1_rand.cuda()
        model2_rand.cuda()
        model1.load_state_dict(torch.load(file1))
        model2.load_state_dict(torch.load(file2))

        print("Getting loaders")
        train_loader, test_loader = Loaders.get_loaders_ffcv(args)

        print("Evaluating accuracies of pretrained models")
        accp = 0.0
        accp = Trainer.evaluate(model1, test_loader)
        model1_acc = accp / 100.0
        print(f"Accuracy of {name1} is {accp}%")
        assert accp > 90

        accp = Trainer.evaluate(model1_rand, test_loader)
        model1_rand_acc = accp / 100.0
        print(f"Accuracy of {name1} random is {accp}%")
        assert accp < 20

        accp = Trainer.evaluate(model2, test_loader)
        model2_acc = accp / 100.0
        print(f"Accuracy of {name2} is {accp}%")
        assert accp > 90

        accp = Trainer.evaluate(model2_rand, test_loader)
        model2_rand_acc = accp / 100.0
        print(f"Accuracy of {name2} random is {accp}%")
        assert accp < 20

        print("Generating table of labels")
        labels: List[List[Tuple[LayerLabel, LayerLabel]]] = \
            LayerLabel.generateTable(
                lambda l1, l2: (l1, l2),
                numbers1,
                numbers2,
        )
        print("Bypassing generating rep shapes")
        print("Bypassing generating stitches")
        print("Generating stitched Nets list (pairs")
        nets: List[Resnet] = [
            # TODO change this name
            ("model1", model1),
            ("model2", model2),
            ("model1_rand", model1_rand),
            ("model2_rand", model2_rand)
        ]

        net_pairs = choose_unordered_subset(nets, 2)

        stitched_nets: Dict[Tuple[str, str], List[List[StitchedResnet]]] = {
            (model1_name, model2_name):
                Table.mappedTable(
                    lambda labels_tuple: StitchedResnet.fromLabels(
                        (model1, model2),
                        *labels_tuple,
                        pool_and_flatten=False,
                    ),
                    labels,
            )
            for (model1_name, model1), (model2_name, model2) in net_pairs
        }

        print("Training stitches")
        vanilla_sims: Dict[Tuple[str, str], List[List[float]]] = {
            (model1_name, model2_name):
            Table.mappedTable(
                stitched_nets_table,
                lambda st: Trainer.train_loop(st))
            for (model1_name, model2_name), stitched_nets_table in stitched_nets.items()
        }

        if not os.path.exists(Experiment.SIMS_FOLDER):
            os.mkdir(Experiment.SIMS_FOLDER)
        for (model1_name, model2_name), vanilla_sim_table in vanilla_sims.items():
            name_no_folder = f"{model1_name}_{model2_name}_sims.pt"
            name_w_folder = os.path.join(
                Experiment.SIMS_FOLDER, name_no_folder)
            torch.save(torch.tensor(vanilla_sim_table), name_w_folder)

        # TODO visualize!


if __name__ == "__main__":
    file_pair = "resnet_1111.pt", "resnet_1111.pt"
    hyps = Hyperparams()
    Experiment.pretrain(hyps)
    Experiment.stitchtrain(hyps, file_pair)
