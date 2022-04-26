
# Enables type annotations using enclosing classes
from __future__ import annotations

from torch.cuda.amp import autocast
import torch.nn.functional as F
import os

from warnings import warn

import unittest

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)

from typing import (
    NoReturn,
    Any,
    Callable,
    Union,
    Dict,
    List,
    Tuple,
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
from visualizer import Visualizer
from cifar import pclone, mapeq, mapneq, flattened_table

def sanity_test_stitches_ptrs(data_ptrs):
    assert len(data_ptrs) > 0, "should have at least some data pointers"
    assert len(data_ptrs) > 1, "should have >1 stitches' data pointers"
    sane = True
    for i in range(len(data_ptrs)):
        for j in range(i+1, len(data_ptrs)):
            sane = sane and data_ptrs[i] != data_ptrs[j]

    return sane

def sanity_test_model_outfrom_into(model, number, loader):
    labels = LayerLabel.labels(number)
    assert len(labels) >= 5
    sane = True
    # For every pair of layers make sure that if you outut from one and input into the next one it works
    for i in range(0, len(labels) - 1):
        out_label = labels[i]
        in_label = labels[i+1]
        this_is_sane = True
        for _input, _ in loader:
            with torch.no_grad():
                with autocast():
                    _intermediate = model.outfrom_forward(_input, out_label)
                    _output = model.into_forward(_intermediate, in_label, pool_and_flatten=True)
                    _exp_output = model(_input)
                    # If the output is not equal to the expected output
                    if (_output == _exp_output).int().min() != 1:
                        sane = False
                        if this_is_sane:
                            warn(f"Sanity failed for {out_label} -> {in_label}")
                            warn(f"Got {_output}\n")
                            warn(f"Expected {_exp_output}\n")
                            warn(f"Cutting off at first occurence for brevity")
                        this_is_sane = False
    return sane

# Basically copied from above, but makes sure that the stitched resnet is not bused
def sanity_test_model_outfrom_into_stitched_resnet(model, number, loader):
    labels = LayerLabel.labels(number)
    assert len(labels) >= 5
    sane = True
    # For every pair of layers make sure that if you outut from one and input into the next one it works
    for i in range(0, len(labels) - 1):
        out_label = labels[i]
        in_label = labels[i+1]
        stitch = Identity()
        stitched_resnet = StitchedResnet(model, model, stitch, out_label, in_label)
        stitched_resnet.freeze()
        this_is_sane = True
        for _input, _ in loader:
            with torch.no_grad():
                with autocast():
                    _output = stitched_resnet(_input)
                    _exp_output = model(_input)
                    # If the output is not equal to the expected output
                    if (_output == _exp_output).int().min() != 1:
                        sane = False
                        if this_is_sane:
                            warn(f"Stitched Resnet:")
                            warn(f"Sanity failed for {out_label} -> {in_label}")
                            warn(f"Got {_output}\n")
                            warn(f"Expected {_exp_output}\n")
                            warn(f"Cutting off at first occurence for brevity")
                        this_is_sane = False
    return sane


class Experiment(object):
    RESNETS_FOLDER = "../../pretrained_resnets/"
    SIMS_FOLDER = "../../sims/"
    HEATMAPS_FOLDER = "../../heatmaps/"

    def __init__(self: Experiment) -> NoReturn:
        pass

    @staticmethod
    def pretrain(args: Any) -> NoReturn:
        """
        Train all resnets of the form[x, y, z, w] where x is 1 or 2, y is 1 or 2, etcetera...
        unless they are already trained (in which case we will use them later). Store them in
        the resnets folder.
        """
        print("Getting FFCV loaders")
        train_loader, test_loader = Loaders.get_loaders_ffcv(args)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        print("Generating combinations")
        combinations = [[1, 1, 1, 1]]

        print(f"Storing resnets in {Experiment.RESNETS_FOLDER}")
        if not os.path.exists(Experiment.RESNETS_FOLDER):
            os.mkdir(Experiment.RESNETS_FOLDER)

        print("Training all combinations")
        for combination in combinations:
            comb_name = "".join(map(str, combination))
            print(f"Training combination {comb_name}")
            x, y, z, w = combination
            fname = os.path.join(Experiment.RESNETS_FOLDER,
                                 f"resnet_{comb_name}.pt")
            if os.path.exists(fname):
                print(f"Skipping {fname}")
                continue
            else:
                # NOT pretrained and NO progress bar (these aren't supported anyways)
                model = ResnetGenerator.generate(
                    f"resnet_{comb_name}",
                    BasicBlock,
                    combination,
                    False,
                    False,
                    num_classes=10,
                )
                model = model.to(device)

                print(
                    f"will train resnet_{comb_name} for {args.epochs} epochs")
                Trainer.train_loop(
                    args,
                    model,
                    train_loader,
                    test_loader,
                )
                acc = Trainer.evaluate(
                    model,
                    test_loader
                )
                print(f"acc was {acc}")

                assert acc < 1.0
                assert acc > 0.8

                torch.save(model.state_dict(), fname)

    @staticmethod
    def stitchtrain(args: Any, filename_pair: Tuple[str, str]) -> NoReturn:
        file1, file2 = filename_pair
        numbers1 = list(map(int, file1.split(".")[0][-4:]))
        numbers2 = list(map(int, file2.split(".")[0][-4:]))
        name1 = "resnet_" + "".join(map(str, numbers1))
        name2 = "resnet_" + "".join(map(str, numbers2))
        file1 = os.path.join(Experiment.RESNETS_FOLDER, file1)
        file2 = os.path.join(Experiment.RESNETS_FOLDER, file2)

        print(f"Experiment {name1} x {name2} in {Experiment.RESNETS_FOLDER}")
        print(f"numbers1: {numbers1}")
        print(f"numbers2: {numbers2}")
        print(f"name1: {name1}")
        print(f"name2: {name2}")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("Loading models")
        pretrained_files = [file1, None, file2, None]
        names = [name1, name1 + "_rand", name2, name2 + "_rand"]
        numbers = [numbers1, numbers1, numbers2, numbers2]
        expected_acc_bounds = [(0.9, 1.0), (0.0, 0.2), (0.9, 1.0), (0.0, 0.2)]
        models = [
            ResnetGenerator.generate(
                name,
                BasicBlock,
                number,
                False,
                False,
                num_classes=10,
            ) for name, number in zip(names, numbers)
        ]

        print(f"Using device {device}")
        for (model, pretrained_file) in zip(models, pretrained_files):
            model.to(device)
            if pretrained_file:
                print(f"Loaded pretrained model {pretrained_file}")
                model.load_state_dict(torch.load(pretrained_file))

        print("Here is A Model (the first one)")
        print(models[0])

        print("Getting loaders for FFCV")
        train_loader, test_loader = Loaders.get_loaders_ffcv(args)

        print("Sanity testing accuracies of pretrained models")
        for name, model, (lo, hi) in zip(names, models, expected_acc_bounds):
            print(f"Asserting that accuracy of {name} is in ({lo}, {hi})")
            acc = Trainer.evaluate(model, test_loader)
            assert acc > lo
            assert acc < hi
        
        print("Sanity testing that the outfrom and into forwards work")
        assert all(map(
            lambda model_number: sanity_test_model_outfrom_into(
                model_number[0],
                model_number[1],
                train_loader,
            ), 
            zip(models, numbers),
        ))

        print("Sanity testing that stitched resnet with stitch=identity works like model")
        assert all(map(
            lambda model_number: sanity_test_model_outfrom_into_stitched_resnet(
                model_number[0],
                model_number[1],
                train_loader,
            ),
            zip(models, numbers),
        ))

        print("Generating table of labels")
        def iden(x, y): return (x, y)
        labels, idx2labels = LayerLabel.generateTable(iden, numbers1, numbers2)
        print("***************** labels *******************")
        pp.pprint(labels)
        print("************** *idx2labels *****************")
        pp.pprint(idx2labels)
        print("********************************************\n")
        print("Sanity checking numbers of labels")
        assert len(idx2labels) == len(labels) * len(labels[0])
        assert LayerLabel.numLayers(numbers1) - 1 == len(labels)
        assert LayerLabel.numLayers(numbers2) == len(labels[0])
        print("Sanity checking that labels grow by 1 along columns")
        assert all(map(
            lambda row: all(((row[i][1] == row[i + 1][1] - 1) for i in range(len(row) - 1))), 
            labels,
        ))
        print("Sanity checking that labels grow by 1 along rows(transposed)")
        assert all(map(
            lambda row: all(((row[i][0] == row[i + 1][0] - 1) for i in range(len(row) - 1))), 
            Table.transposed(labels),
        ))

        print("Generating pairs of networks to stitch")
        named_models = list(zip(names, models))
        pairs = [(named_models[0], named_models[2])]
        if args.control or args.control:
            raise NotImplementedError("Only stitching model1 with model2")

        print("Generating stitches for each pair of layers for each pair of models")
        assert len(pairs) == 1, "Does not support stitching more than one pair yet"
        stitched_nets: Dict[Tuple[str, str], List[List[StitchedResnet]]] = {
            (model1_name, model2_name):
                Table.mappedTable(
                    lambda labels_tuple: StitchedResnet.fromLabels(
                        (model1, model2),
                        labels_tuple[0],
                        labels_tuple[1],
                    ).to(device),
                    labels,
            )
            for (model1_name, model1), (model2_name, model2) in pairs
        }
        print("***************** labels *******************")
        for sn_name, sn_table in stitched_nets.items():
            print(f"{sn_name}")
            for row in sn_table:
                for sn in row:
                    print(f"{sn.send_label} -> {sn.recv_label}")
                    print(sn.stitch)
                    print("")
        print("********************************************\n")

        print("Generating debugging tests to make sure that models weights did NOT change")
        DEBUG_ORIG_MODELS_PARAMS =\
            {
                name : pclone(model)\
                for name, model in named_models
            }
        DEBUG_ORIG_STITCHES_PARAMS =\
            {
                # Table of stitches -> Table of lists of torch.Tensors -> List of Lists of torch.Tensors
                # -> List of torch.Tensors.
                model_pair : flattened_table(flattened_table(
                    Table.mappedTable(pclone, table),
                ))\
                for model_pair, table in stitched_nets.items()
            }
        DEBUG_ORIG_DATA_PTRS = {
            # Table of stitches -> table of lists if data pointers for each param -> list of
            # lists of data pointers -> list of data pointers
            model_pair : flattened_table(flattened_table(
                Table.mappedTable(lambda st: list(map(lambda p: p.data_ptr(), st.parameters())), table),
            ))\
            for model_pair, table in stitched_nets.items()
        }
        
        # Check that for each model the data pointers are unique for each stitch
        assert all(map(sanity_test_stitches_ptrs, DEBUG_ORIG_DATA_PTRS.values()))
        assert len(DEBUG_ORIG_DATA_PTRS.keys()) == 1, "does not yet supported > 1 model"
        # IN the future when we have more models, we will want to check that the pointers are
        # NOT the same across models but ARE the same within models

        # Debugging the debugger
        assert len(list(DEBUG_ORIG_MODELS_PARAMS.items())) > 0
        assert type(list(DEBUG_ORIG_MODELS_PARAMS.items())[0][1]) == list
        assert len(list(DEBUG_ORIG_MODELS_PARAMS.items())[0][1]) > 0
        assert type(list(DEBUG_ORIG_MODELS_PARAMS.items())[0][1][0]) == torch.Tensor

        print(f"There are {len(idx2labels)} pairs")
        def train_with_info(st: StitchedResnet):
            print(f"\tTraining on stitch {st.send_label} -> {st.recv_label}")
            st.freeze()
            acc = Trainer.train_loop(args, st, train_loader, test_loader)# \
            # if st.send_label.isBlock() and \
            # st.send_label.getBlockset() in [1, 2, 3, 4] \
            # and st.recv_label - 1 == st.send_label \
            # else 0.0
            print(f"\tGot acc {acc}")
            return acc

        print("Training stitches")
        vanilla_sims: Dict[Tuple[str, str], List[List[float]]] = {
            (model1_name, model2_name):
            Table.mappedTable(
                train_with_info,
                stitched_nets_table,
            )
            for (model1_name, model2_name), stitched_nets_table in stitched_nets.items()
        }

        print("Creating sims and heatmaps folder if necessary")
        if not os.path.exists(Experiment.SIMS_FOLDER):
            os.mkdir(Experiment.SIMS_FOLDER)
        if not os.path.exists(Experiment.HEATMAPS_FOLDER):
            os.mkdir(Experiment.HEATMAPS_FOLDER)
        
        print("Saving sims and heatmaps")
        for (model1_name, model2_name), vanilla_sim_table in vanilla_sims.items():
            sim_path = os.path.join(
                Experiment.SIMS_FOLDER, f"{model1_name}_{model2_name}_sims.pt")
            heat_path = os.path.join(
                Experiment.HEATMAPS_FOLDER, f"{model1_name}_{model2_name}_heatmaps.png")
            print(f"Saving tensor to {sim_path}")
            torch.save(torch.tensor(vanilla_sim_table), sim_path)

            # Note might be nice to not have to save and then re-load
            print(f"Visualizing at {heat_path}")
            Visualizer.matrix_heatmap(sim_path, heat_path)

        print("Sanity testing that models weights did NOT change and stitches' weights did")
        DEBUG_NEW_MODELS_PARAMS =\
            {
                name : pclone(model)\
                for name, model in named_models
            }
        DEBUG_NEW_STITCHES_PARAMS =\
            {
                # Table of stitches -> Table of lists of torch.Tensors -> List of Lists of torch.Tensors
                # -> List of torch.Tensors.
                model_pair : flattened_table(flattened_table(
                    Table.mappedTable(pclone, table),
                ))\
                for model_pair, table in stitched_nets.items()
            }

        # NO models should have changed and ALL stitches should have changed
        assert mapeq(DEBUG_ORIG_MODELS_PARAMS, DEBUG_NEW_MODELS_PARAMS)
        assert mapneq(DEBUG_ORIG_STITCHES_PARAMS, DEBUG_NEW_STITCHES_PARAMS)

        print("Sanity testing that model has retained good accuracy")
        assert Trainer.evaluate(models[0], train_loader) > 0.9
        print("OK!")

if __name__ == "__main__":
    file_pair = "resnet_1111.pt", "resnet_1111.pt"
    hyps = Hyperparams()
    hyps.epochs = 40
    Experiment.pretrain(hyps)
    hyps.epochs = 10 # NOTE should be bigger
    Experiment.stitchtrain(hyps, file_pair)
