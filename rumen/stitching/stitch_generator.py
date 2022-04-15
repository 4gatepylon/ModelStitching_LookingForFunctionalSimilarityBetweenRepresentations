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

import torch.nn as nn

from layer_label import LayerLabel
from rep_shape import RepShape
from stitched_resnet import StitchedResnet

# https://docs.python.org/3/library/typing.html#typing.ParamSpec
T = TypeVar('T')
P = ParamSpec('P')


class StitchGenerator(object):
    """ Class to generate stitches between two layers with given shapes """

    USE_BIAS: bool = True

    def __init__(self: StitchGenerator, shape: Tuple(RepShape, RepShape)) -> NoReturn:
        self.send_shape, self.recv_shape = shape

    def generate(self: StitchGenerator) -> nn.Module:
        if self.send_shape.isVector():
            raise ValueError(
                f"Cannot send from vector layer {self.send_shape}")
        else:
            if self.recv_shape.isVector():
                return nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(self.send_shape.numActivations(),
                              self.recv_shape.numActivations()),
                )
            else:
                # Recall that depth always doubles and width/height each always halve
                send_depth = self.send_shape.depth()
                recv_depth = self.recv_shape.depth()
                if recv_depth <= send_depth:
                    ratio = send_depth // recv_depth
                    return nn.Conv2d(send_depth, recv_depth, ratio, stride=ratio, bias=StitchGenerator.USE_BIAS)
                else:
                    ratio = recv_depth // send_depth
                    return nn.Sequential(
                        nn.Upsample(scale_factor=ratio, mode='nearest'),
                        nn.Conv2d(send_depth, recv_depth, 1, stride=1, bias=True))


class TestStitchGenerator(unittest.TestCase):
    def test_nothing(self: TestStitchGenerator) -> NoReturn:
        raise NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=2)
