# Enables type annotations using enclosing classes
from __future__ import annotations

import unittest

# Enables more interesting type annotations
from typing_extensions import (
    ParamSpec,
)
from typing import (
    Dict,
    NoReturn,
    List,
    Tuple,
    TypeVar,
)

import torch.nn as nn

from rep_shape import RepShape

# https://docs.python.org/3/library/typing.html#typing.ParamSpec
T = TypeVar('T')
P = ParamSpec('P')


class StitchGenerator(object):
    """
    Class to generate stitches between two layers with given shapes.
    It generates plain stitches. That is to say it does NOT create stitched
    networks. To create a stitched network you need to use the StitchedResnet
    object.
    """

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
                # so the greater of the two heights corresponds to the earlier layer.
                # NOTE It is cleaner to use height because depth might be 3.
                send_height = self.send_shape.height()
                send_depth = self.send_shape.depth()
                recv_height = self.recv_shape.height()
                recv_depth = self.recv_shape.depth()
                if recv_height <= send_height:
                    ratio = send_height // recv_height
                    return nn.Conv2d(
                        send_depth,
                        recv_depth,
                        ratio,
                        stride=ratio,
                        bias=StitchGenerator.USE_BIAS,
                    )
                else:
                    ratio = send_height // recv_height
                    return nn.Sequential(
                        nn.Upsample(
                            scale_factor=ratio,
                            mode='nearest',
                        ),
                        nn.Conv2d(
                            send_depth,
                            recv_depth,
                            1,
                            stride=1,
                            bias=True,
                        ),
                    )


class TestStitchGenerator(unittest.TestCase):
    # NOTE: we do not test that there is bias or not (this is a design choice)

    def test_tensor2tensor(self: TestStitchGenerator) -> NoReturn:
        # Downsample example
        tensor1: RepShape = RepShape((3, 32, 32))
        tensor2: RepShape = RepShape((128, 16, 16))
        stitch12: nn.Module = StitchGenerator((tensor1, tensor2)).generate()

        # Upsample example
        tensor3: RepShape = RepShape((512, 4, 4))
        tensor4: RepShape = RepShape((64, 32, 32))
        stitch34: nn.Module = StitchGenerator((tensor3, tensor4)).generate()

        # Test the downsampling exmaple
        self.assertEqual(type(stitch12), nn.Conv2d)
        self.assertEqual(stitch12.in_channels, 3)
        self.assertEqual(stitch12.out_channels, 128)
        self.assertEqual(stitch12.kernel_size, (2, 2))
        self.assertEqual(stitch12.stride, (2, 2))

        # Make sure that the defaults are used because they are necessary to maintain the shape
        self.assertEqual(stitch12.padding, (0, 0))
        self.assertEqual(stitch12.padding_mode, 'zeros')
        self.assertEqual(stitch12.dilation, (1, 1))
        self.assertEqual(stitch12.groups, 1)

        # Test the upsampling example
        self.assertEqual(type(stitch34), nn.Sequential)
        self.assertEqual(len(stitch34), 2)
        self.assertEqual(type(stitch34[0]), nn.Upsample)
        self.assertEqual(type(stitch34[1]), nn.Conv2d)
        self.assertEqual(stitch34[1].in_channels, 512)
        self.assertEqual(stitch34[1].out_channels, 64)
        self.assertEqual(stitch34[1].kernel_size, (1, 1))
        self.assertEqual(stitch34[1].stride, (1, 1))

    def test_tensor2vector(self: TestStitchGenerator) -> NoReturn:
        tensor: RepShape = RepShape((3, 64, 64))
        vector: RepShape = RepShape(10)
        stitch: nn.Module = StitchGenerator((tensor, vector)).generate()

        self.assertEqual(type(stitch), nn.Sequential)
        self.assertEqual(len(stitch), 2)
        self.assertEqual(type(stitch[0]), nn.Flatten)
        self.assertEqual(type(stitch[1]), nn.Linear)
        self.assertEqual(stitch[1].in_features, 3 * 64 * 64)
        self.assertEqual(stitch[1].out_features, 10)

    def test_vector2tensor(self: TestStitchGenerator) -> NoReturn:
        vector: RepShape = RepShape(10)
        tensor: RepShape = RepShape((3, 64, 64))

        self.assertRaises(Exception, StitchGenerator(
            (vector, tensor)).generate)


if __name__ == '__main__':
    unittest.main(verbosity=2)
