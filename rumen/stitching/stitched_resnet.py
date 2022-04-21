# Enables type annotations using enclosing classes
from __future__ import annotations

import unittest

# Enables more interesting type annotations
from typing_extensions import (
    Concatenate,
    ParamSpec,
)
from typing import (
    Dict,
    NoReturn,
    Callable,
    Union,
    List,
    Optional,
    Tuple,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

# A little bit of a hack for ffcv
from torch.cuda.amp import autocast

from resnet.resnet import Resnet
from layer_label import LayerLabel
from rep_shape import RepShape
from stitch_generator import StitchGenerator
from trainer import Trainer, Hyperparams
from loaders import MockDataLoader
from cifar import pclone, listeq


class StitchedResnet(nn.Module):
    """
    A StitchedResnet is a wrapper to enable easy manipulation of the stitched
    resnet which is created by taking the first N layers of one model (the prefix), outputting
    their intermediate representation into a stitch, then inputting that into the last M
    layers of another model (the suffix).

    NOTE that you require that the sender and reciever have, respectively, outfrom_forward, and
    into_forward methods. Otherwise this will fail!
    """

    def __init__(self, sender: nn.Module, reciever: nn.Module, stitch: nn.Module, send_label: LayerLabel, recv_label: LayerLabel) -> None:
        super(StitchedResnet, self).__init__()
        self.sender = sender
        self.reciever = reciever
        self.send_label = send_label
        self.recv_label = recv_label
        self.stitch = stitch

    def parameters(self: StitchedResnet) -> List[torch.nn.Parameter]:
        # Sender and reciever should NOT be optimized
        return self.stitch.parameters()

    def freeze_sender(self: StitchedResnet) -> NoReturn:
        for p in self.sender.parameters():
            p.requires_grad = False

    def freeze_reciever(self: StitchedResnet) -> NoReturn:
        for p in self.reciever.parameters():
            p.requires_grad = False

    def train(self: StitchedResnet) -> NoReturn:
        self.stitch.train()
        self.sender.eval()
        self.reciever.eval()

    def eval(self: StitchedResnet) -> NoReturn:
        self.stitch.eval()
        self.sender.eval()
        self.reciever.eval()

    def freeze(self: StitchedResnet) -> NoReturn:
        self.freeze_sender()
        self.freeze_reciever()

    def forward(self: StitchedResnet, x: torch.Tensor) -> torch.Tensor:
        # Pool and flatten should only happen for MSE losses (otherwise we don't, since
        # we want the full tensor `representation`)
        with autocast():
            h = self.sender.outfrom_forward(
                x, self.send_label, pool_and_flatten=False)
            # print(f"\t\t\tstitch gets shape {h.shape}")
            h = self.stitch(h)
            # print("\t\t\t\done")
            h = self.reciever.into_forward(
                h, self.recv_label, pool_and_flatten=self.recv_label.isFc())
        return h

    @staticmethod
    def fromLabels(
            nets: Tuple[Resnet, Resnet],
            send_label: LayerLabel,
            recv_label: LayerLabel,
            pool_and_flatten: Optional[bool] = False) -> StitchedResnet:
        #print(f"send label is {send_label} and recv_label is {recv_label}")
        shapes = RepShape.stitchShapes(send_label, recv_label)
        stitch: nn.Module = StitchGenerator.generate(shapes)
        return StitchedResnet(nets[0], nets[1], stitch, send_label, recv_label)


class MockResnet(nn.Module):
    """ Mock Resnet that is designed to be faster than a regular Resnet for unit testing. """

    def __init__(self: MockResnet):
        super().__init__()
        # input -> conv1 -> flatten -> layer (all blocks/blocksets) -> fc -> output
        self.conv1 = nn.Conv2d(3, 1, kernel_size=32,
                               stride=1, padding=0, bias=False)
        self.layer = nn.Linear(1, 1)
        self.fc = nn.Linear(1, 10)

    def outfrom_forward(
        self: MockResnet,
        x: torch.Tensor,
        vent: LayerLabel,
        pool_and_flatten: bool = False,
    ) -> torch.Tensor:
        if vent.isInput():
            return x
        elif vent.isConv1():
            return self.conv1(x)
        elif vent.isFc():
            return self.forward(x)
        elif vent.isOutput():
            raise Exception("Can't vent outfrom output")
        else:
            return self.layer(torch.flatten(self.conv1(x), 1))

    def into_forward(
        self: MockResnet,
        x: torch.Tensor,
        vent: LayerLabel,
    ) -> torch.Tensor:
        if vent.isInput():
            raise Exception("Can't go into input in into_forward")
        elif vent.isConv1():
            return self.forward(x)
        elif vent.isFc():
            return self.fc(x)
        elif vent.isOutput():
            return x
        else:
            return self.fc(self.layer(x))

    def forward(
        self: MockResnet,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.fc(self.layer(torch.flatten(self.conv1(x))))


class TestStitchedResnet(unittest.TestCase):
    """
    Test that the stitched resnet works as intended. Specifically,
    and most importantly, the prefix and suffix should both be
    frozen during training, while the stitch should NOT be frozen.
    """

    # TODO test these options
    # 1. FFCV dataloader
    # 2. CIFAR dataloader
    # 3. full model (real resnet)
    # 4. number of parameters is right
    # 5. pointers are different for different stitches in the stitching table

    def test_proper_freeze_mock(self: TestStitchedResnet) -> NoReturn:
        R = [1, 1, 1, 1]

        # Create a mock resnet
        prefix = MockResnet()
        suffix = MockResnet()

        # Try stitching the linear layers
        output_shape = RepShape((1, 1, 1))
        output_label = LayerLabel(LayerLabel.CONV1, R)
        input_shape = RepShape(1)
        input_label = LayerLabel((1, 0), R)
        stitch = StitchGenerator.generate((output_shape, input_shape))

        stitched = StitchedResnet(
            prefix, suffix, stitch, output_label, input_label)
        stitched.freeze()
        args = Hyperparams.forTesting()

        train_loader, test_loader = MockDataLoader.mock_loaders(args)

        # Make sure that stitched network yields the right parameters so that
        # we can pass it as a black box "nn.Module"
        stitched_params = pclone(stitched)
        should_be_stitched_params = pclone(stitched.stitch)
        self.assertTrue(listeq(stitched_params, should_be_stitched_params))

        # Get the params before training
        sender_params = pclone(stitched.sender)
        reciever_params = pclone(stitched.reciever)

        # Train for an epoch
        acc = Trainer.train_loop(args, stitched, train_loader, test_loader)
        # self.assertTrue(acc > 0.0)
        # self.assertTrue(acc < 1.0)

        # Get the parameters after training
        new_reciever_params = pclone(stitched.reciever)
        new_sender_params = pclone(stitched.sender)
        new_stitched_params = pclone(stitched)

        # Make sure that the stitch updated (learned) but that the prefix and suffix
        # did NOT update (they should be frozen)
        self.assertFalse(listeq(new_stitched_params, stitched_params))
        self.assertTrue(listeq(new_sender_params, sender_params))
        self.assertTrue(listeq(new_reciever_params, reciever_params))

    @ unittest.skip("Unimplemented, but this would run too slow without a GPU regardless.")
    def test_proper_freeze_full(self: TestStitchedResnet) -> NoReturn:
        """
        This big test (requires GPU) uses two real resnets to test whether the stitched resnet
        is behaving as intended. It will create a table of all possible stitches for the pair of
        resnets below (chosen so that there would be shared numbers of blocks for some blocksets,
        different numbers of blocks for some other blocksets, and cases with more than one block
        per blockset).

        NOTE this covers downsampling, regular stitching, upsampling, AND flattened sttiching.
        """
        R1: List[int] = [1, 1, 2, 1]
        R2: List[int] = [1, 2, 2, 1]
        raise NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=2)
