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
from defusedxml import NotSupportedError

import torch
import torch.nn as nn

from resnet.resnet import Resnet
from layer_label import LayerLabel
from rep_shape import RepShape
from stitch_generator import StitchGenerator


class StitchedResnet(nn.Module):
    """
    A StitchedResnet is a wrapper to enable easy manipulation of the stitched
    resnet which is created by taking the first N layers of one model (the prefix), outputting
    their intermediate representation into a stitch, then inputting that into the last M
    layers of another model (the suffix).
    """

    ###########################################################################
    class ResnetSegment(nn.Module):
        """ Superclass to prefixes and suffixes """

        def __init__(self: StitchedResnet.ResnetSegment, net: Resnet, outfrom: LayerLabel) -> NoReturn:
            super().__init__()
            self.net = net
            self.outfrom = outfrom
            self.frozen = False
            self.freeze()

        def parameters(self: StitchedResnet.ResnetSegment) -> List[torch.Tensor]:
            raise NotSupportedError(
                f"ResnetSegiment {self} should not have its parameters used")

        def freeze(self: StitchedResnet.ResnetSegment):
            self.frozen = True
            for p in self.net.parameters():
                p.requires_grad = False

        def forward(self: StitchedResnet.ResnetSegment, x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError("Called .forward of ResnetSegment")

    class ResnetPrefix(ResnetSegment):
        """ Keep state about prefix and enable a simple forward method (args same as ResnetSegment) """

        def __init__(self: StitchedResnet.ResnetPrefix, *args: ..., pool_and_flatten: Optional[bool] = False) -> NoReturn:
            super().__init__(*args)
            self.pool_and_flatten = pool_and_flatten

        def forward(self: StitchedResnet.ResnetPrefix, x: torch.Tensor) -> torch.Tensor:
            return self.net.outfrom_forward(x, self.outfrom, pool_and_flatten=self.pool_and_flatten)

    class ResnetSuffix(ResnetSegment):
        """ Keep state about suffix and enable a simple forward method (args same as ResnetSegment) """

        def __init__(self: StitchedResnet.ResnetSuffix, *args: ...):
            super().__init__(*args)

        def forward(self: StitchedResnet.ResnetSuffix, x: torch.Tensor) -> torch.Tensor:
            return self.net.into_forward(x, self.into)
    ###########################################################################

    def __init__(self, sender: ResnetPrefix, reciever: ResnetSuffix, stitch: nn.Module):
        super(StitchedResnet, self).__init__()
        self.sender = sender
        self.reciever = reciever
        self.send_label = StitchedResnet.ResnetPrefix.outfrom
        self.recv_label = StitchedResnet.ResnetSuffix.into
        self.stitch = stitch

    def freeze_sender(self: StitchedResnet) -> NoReturn:
        self.sender.freeze()

    def freeze_reciever(self: StitchedResnet) -> NoReturn:
        self.reciever.freeze()

    def freeze(self: StitchedResnet) -> NoReturn:
        self.sender.freeze()
        self.reciever.freeze()

    def forward(self: StitchedResnet, x: torch.Tensor) -> torch.Tensor:
        h = self.sender(x)
        h = self.stitch(h)
        h = self.reciever(h)
        return h

    def frozen(self: StitchedResnet) -> StitchedResnet:
        self.freeze()
        return self

    @staticmethod
    def fromLabels(
            nets: Tuple[Resnet, Resnet],
            send_label: LayerLabel,
            recv_label: LayerLabel,
            pool_and_flatten: Optional[bool] = False) -> StitchedResnet:
        shapes = RepShape.stitchShapes(send_label, recv_label)
        prefix: StitchedResnet.ResnetPrefix = StitchedResnet.ResnetPrefix(
            nets[0],
            shapes[0],
            pool_and_flatten=pool_and_flatten,
        )
        suffix: StitchedResnet.ResnetSuffix = StitchedResnet.ResnetSuffix(
            nets[1],
            shapes[1],
        )
        stitch: nn.Module = StitchGenerator(shapes).generate()
        return StitchedResnet(prefix, suffix, stitch)

# TODO the only test we really need here is that freezing works


class TestStitchedResnet(unittest.TestCase):
    """
    Test that the stitched resnet works as intended. Specifically,
    and most importantly, the prefix and suffix should both be
    frozen during training, while the stitch should NOT be frozen.
    """

    # These are outside of the MockResnet because they are used elsewhere to test
    MOCK_IN_DIM: int = 2
    MOCK_OUT_DIM: int = 2

    class MockResnet(Resnet):
        """ Mock Resnet that is designed to be faster than a regular Resnet for unit testing. """

        MOCK_R: List[int] = [1, 1, 1, 1]
        MOCK_LABEL: LayerLabel = LayerLabel(LayerLabel.CONV1, MOCK_R)

        def __init__(self: TestStitchedResnet.MockResnet):
            # NOTE: do NOT call super() because we are creating a mock
            # for the purposes of enabling fast unit testing. We use a list
            # because it avoids using the nn.Module's __setattr__ method's
            # side effects on the `_modules` dictionary (read here:
            # https://stackoverflow.com/questions/63058355/why-is-the-super-constructor-necessary-in-pytorch-custom-modules)
            # (Basically, it's a hack.)
            self.layers = [None]
            self.layers[0] = layer = nn.Linear(
                TestStitchedResnet.MOCK_IN_DIM,
                TestStitchedResnet.MOCK_OUT_DIM,
            )

        def layer(self: TestStitchedResnet.MockResnet) -> nn.Module:
            return self.layers[0]

        def outfrom_forward(
            self: TestStitchedResnet.MockResnet,
            x: torch.Tensor,
            vent: LayerLabel,
            pool_and_flatten: bool = False,
        ) -> torch.Tensor:
            # Coming out from the layer (imagine this is a prefix)
            return self.layer()(x)

        def into_forward(
            self: TestStitchedResnet.MockResnet,
            x: torch.Tensor,
            vent: LayerLabel,
        ) -> torch.Tensor:
            # Going into the Layer (imagine this is a suffix)
            return self.layer()(x)

        def forward(
            self: TestStitchedResnet.MockResnet,
            x: torch.Tensor,
        ) -> torch.Tensor:
            # Going through the layer (imagine this is a regular network)
            return self.layer()(x)

        def parameters(self: TestStitchedResnet.MockResnet) -> List[torch.Tensor]:
            return self.layer().parameters()

    # These methods are just to help test Prefixes easier
    def MockResnetPrefix(self: TestStitchedResnet) -> StitchedResnet.ResnetPrefix:
        return StitchedResnet.ResnetPrefix(
            TestStitchedResnet.MockResnet(),
            TestStitchedResnet.MockResnet.MOCK_LABEL,
        )

    def MockResnetSuffix(self: TestStitchedResnet) -> StitchedResnet.ResnetSuffix:
        return StitchedResnet.ResnetSuffix(
            TestStitchedResnet.MockResnet(),
            TestStitchedResnet.MockResnet.MOCK_LABEL,
        )

    def test_prefix_parameters(self: TestStitchedResnet) -> NoReturn:
        self.assertRaises(Exception, self.MockResnetPrefix().parameters)

    def test_suffix_parameters(self: TestStitchedResnet) -> NoReturn:
        self.assertRaises(Exception, self.MockResnetSuffix().parameters)
    ###########################################################################

    def test_proper_freeze_mock_flatten(self: TestStitchedResnet) -> NoReturn:
        raise NotImplementedError

    def test_proper_freeze_mock_downsample(self: TestStitchedResnet) -> NoReturn:
        raise NotImplementedError

    def test_proper_freeze_mock_upsample(self: TestStitchedResnet) -> NoReturn:
        raise NotImplementedError

    @unittest.skip("Unimplemented, but this would run too slow without a GPU regardless.")
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
