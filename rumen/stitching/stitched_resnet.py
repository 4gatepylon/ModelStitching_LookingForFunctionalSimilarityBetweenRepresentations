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
    TypeVar,
)

import torch
import torch.nn as nn

from resnet import Resnet
from layer_label import LayerLabel
from rep_shape import RepShape
from stitch_generator import StitchGenerator

# https://docs.python.org/3/library/typing.html#typing.ParamSpec
T = TypeVar('T')
P = ParamSpec('P')


class ResnetPrefix(nn.Module):
    def __init__(self: ResnetPrefix, net: Resnet, outfrom: LayerLabel, pool_and_flatten: Optional[bool] = False) -> NoReturn:
        self.net = net
        self.outfrom = outfrom
        self.pool_and_flatten = pool_and_flatten
        self.frozen = False
        self.freeze()

    def parameters(self: ResnetPrefix) -> List[torch.Tensor]:
        raise NotImplementedError

    def freeze(self: ResnetPrefix) -> NoReturn:
        self.frozen = True
        for p in self.net.parameters():
            p.requires_grad = False

    def forward(self: ResnetPrefix, x: torch.Tensor) -> torch.Tensor:
        return self.net.outfrom_forward(x, self.outfrom, pool_and_flatten=self.pool_and_flatten)


class ResnetSuffix(nn.Module):
    def __init__(self: ResnetSuffix, net: Resnet, into: LayerLabel):
        self.net = net
        self.into = into
        self.frozen = False
        self.freeze()

    def parameters(self: ResnetPrefix) -> List[torch.Tensor]:
        raise NotImplementedError

    def freeze(self: ResnetPrefix) -> NoReturn:
        self.frozen = True
        for p in self.net.parameters():
            p.requires_grad = False

    def forward(self: ResnetSuffix, x: torch.Tensor) -> torch.Tensor:
        return self.net.into_forward(x, self.into)


class StitchedResnet(nn.Module):
    def __init__(self, sender: ResnetPrefix, reciever: ResnetSuffix, stitch: nn.Module):
        super(StitchedResnet, self).__init__()
        self.sender = sender
        self.reciever = reciever
        self.send_label = ResnetPrefix.outfrom
        self.recv_label = ResnetSuffix.into
        self.stitch = stitch

    def freeze_sender(self: StitchedResnet) -> NoReturn:
        self.sender.freeze()

    def freeze_reciever(self: StitchedResnet) -> NoReturn:
        self.reciever.freeze()

    def freeze(self: StitchedResnet) -> NoReturn:
        self.sender.freeze()
        self.reciever.freeze()

    def forward(self, x):
        h = self.sender(x)
        h = self.stitch(h)
        h = self.reciever(h)
        return h

    @staticmethod
    def fromLabels(
            nets: Tuple[Resnet, Resnet],
            send_label: LayerLabel,
            recv_label: LayerLabel,
            pool_and_flatten: Optional[bool] = False) -> StitchedResnet:
        shapes = RepShape.stitchShapes(send_label, recv_label)
        prefix: ResnetPrefix = ResnetPrefix(
            nets[0], shapes[0], pool_and_flatten=pool_and_flatten)
        suffix: ResnetSuffix = ResnetSuffix(
            nets[1], shapes[1])
        stitch: nn.Module = StitchGenerator(shapes).generate()
        return StitchedResnet(prefix, suffix, stitch)


class TestResnetPrefix(unittest.TestCase):
    def test_nothings(self):
        raise NotImplementedError


class TestResnetSuffix(unittest.TestCase):
    def test_nothings(self):
        raise NotImplementedError


class TestStitchedResnet(unittest.TestCase):
    def test_nothing(self: TestStitchedResnet) -> NoReturn:
        raise NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=2)
