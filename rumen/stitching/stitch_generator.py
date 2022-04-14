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

# TODO refactor this into the class


def resnet18_34_stitch(snd_shape, rcv_shape):
    if type(snd_shape) == int:
        raise Exception("can't send from linear layer")

    snd_depth, snd_hw, _ = snd_shape
    if type(rcv_shape) == int:
        # you can pass INTO an fc
        # , dtype=torch.float16))
        return nn.Sequential(nn.Flatten(), nn.Linear(snd_depth * snd_hw * snd_hw, rcv_shape))

    # else its tensor to tensor
    rcv_depth, rcv_hw, _ = rcv_shape
    upsample_ratio = rcv_hw // snd_hw
    downsample_ratio = snd_hw // rcv_hw

    # Downsampling (or same size: 1x1) is just a strided convolution since size decreases always by a power of 2
    # every set of blocks (blocks are broken up into sets that, within those sets, have the same size).
    if downsample_ratio >= upsample_ratio:
        # print(f"DOWNSAMPLE {snd_shape} -> {rcv_shape}: depth={snd_depth} -> {rcv_depth}, kernel_width={downsample_ratio}, stride={downsample_ratio}")
        # , dtype=torch.float16)
        return nn.Conv2d(snd_depth, rcv_depth, downsample_ratio, stride=downsample_ratio, bias=True)
    else:
        return nn.Sequential(
            nn.Upsample(scale_factor=upsample_ratio, mode='nearest'),
            nn.Conv2d(snd_depth, rcv_depth, 1, stride=1, bias=True))  # , dtype=torch.float16))


class StitchGenerator(object):
    def __init__(self: StitchGenerator) -> NoReturn:
        raise NotImplementedError

    def shape(self: StitchGenerator, send: LayerLabel, recv: LayerLabel) -> RepShape:
        raise NotImplementedError

    def generate(self: StitchGenerator) -> nn.Module:
        raise NotImplementedError


class StitchGeneratorTest(unittest.TestCase):
    def test_nothing(self: StitchGeneratorTest) -> NoReturn:
        raise NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=2)
