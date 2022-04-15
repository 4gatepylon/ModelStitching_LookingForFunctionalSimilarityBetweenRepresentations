# Enables type annotations using enclosing classes
from __future__ import annotations
from multiprocessing.reduction import recv_handle

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

from layer_label import LayerLabel

# https://docs.python.org/3/library/typing.html#typing.ParamSpec
T = TypeVar('T')
P = ParamSpec('P')


class RepShape(object):
    """
    RepShape is a simple wrapper for (int, int, int) | int, where it describes
    the shape of a representation. A representation for us is either a tensor
    with (depth, height, width) or simply a vector with a single dimension.
    We ignore the batch dimension.
    """

    NUM_CLASSES: int = 10

    def __init__(self: RepShape, shape: Union[Tuple[int, int, int], int]) -> NoReturn:
        self.shape = shape

    def checkValid(func: Callable[Concatenate[RepShape, P], T]) -> Callable[Concatenate[RepShape, P], T]:
        """
        Simply checks that this is a valid RepShape. It must be either an integer or a tuple of three
        integers which must all be non-negative since they are the dimensions of a tensor.
        """

        def wrapper(self: RepShape, *args: P.args, **kwargs: P.kwargs) -> T:
            if type(self.shape) != int and type(self.shape) != tuple:
                # Must be either an int or a tuple of ints
                raise Exception(f"Invalid shape: {self.shape}")
            elif type(self.shape) == tuple:
                # If it's a tuple it must have 3 ints all that are positive
                if len(self.shape) != 3:
                    raise Exception(
                        f"Only supports tensor shapes of length 3, found shape {self.shape}")
                if min(self.shape) <= 0:
                    raise Exception(
                        f"Shape must be strictly positive in all dimensions, found {self.shape}")
                if tuple(map(lambda x: int(x), self.shape)) != self.shape:
                    raise Exception(
                        f"Shape must be integers, found {self.shape}")

            return func(self, *args, **kwargs)
        return wrapper

    @checkValid
    def isTensor(self: RepShape):
        return type(self.shape) == tuple

    @checkValid
    def depth(self: RepShape) -> int:
        assert self.isTensor()
        return self.shape[0]

    @checkValid
    def width(self: RepShape) -> int:
        # Width is meant to be overloaded to work with both types of reps
        return self.shape[2] if self.isTensor() else self.shape

    @checkValid
    def height(self: RepShape) -> int:
        assert self.isTensor()
        return self.shape[1]

    @checkValid
    def isVector(self: RepShape):
        return type(self.shape) == int

    @checkValid
    def numActivations(self: RepShape) -> int:
        if self.isVector():
            return self.shape
        else:
            return self.shape[0] * self.shape[1] * self.shape[2]

    @checkValid
    def __repr__(self: RepShape) -> str:
        return repr(self.shape)

    @checkValid
    def __hash__(self: RepShape) -> int:
        return hash(self.shape)

    @checkValid
    def __eq__(self: RepShape, other: object) -> bool:
        return type(other) == RepShape and self.shape == other.shape

    @checkValid
    def __str__(self: RepShape) -> str:
        return repr(self)

    @staticmethod
    def label2outputRepShape(label: LayerLabel) -> RepShape:
        if label.isInput():
            return RepShape((3, 32, 32))
        if label.isConv1():
            return RepShape((64, 32, 32))
        elif label.isBlock():
            blockset: int = label.getBlockset()
            base_depth: int = 64
            base_width_height: int = 32
            # NOTE that in ResNets the depth always doubles every layer and always halves every layer
            multiplier: int = 2**(blockset - LayerLabel.BLOCKSET_MIN)
            depth: int = base_depth * multiplier
            height: int = base_width_height // multiplier
            return RepShape((depth, height, height))
        elif label.isFc():
            # NOTE: that right now we only support classification for CIIFAR
            return RepShape(RepShape.NUM_CLASSES)
        elif label.isOutput():
            raise Exception(f"Output layer shape is not supported")
        else:
            raise Exception(f"Unknown label type: {label}")

    @staticmethod
    def stitchShapes(send: LayerLabel, recv: LayerLabel) -> Tuple[RepShape, RepShape]:
        # NOTE that the shape the reciever should get is that which the layer before it outputs
        send_shape: RepShape = RepShape.label2outputRepShape(send)
        recv_shape: RepShape = RepShape.label2outputRepShape(recv - 1)
        return (send_shape, recv_shape)


class TestRepShape(unittest.TestCase):
    """ Simple unit tester to sanity check RepShape """

    def test_checkValid(self: TestRepShape) -> NoReturn:
        # NOTE: vscode gets confused and thinks this is unreachable code, but it's not
        zeroTensor: RepShape = RepShape((3, 4, 5))
        zeroVector: RepShape = RepShape(3)
        negTensor: RepShape = RepShape((3, -4, 5))
        negVector: RepShape = RepShape(-3)
        nonIntTensor: RepShape = RepShape((3, 4, 3.3))

        self.assertRaises(Exception, zeroTensor.checkValid())
        self.assertRaises(Exception, zeroVector.checkValid())
        self.assertRaises(Exception, negTensor.checkValid())
        self.assertRaises(Exception, negVector.checkValid())
        self.assertRaises(Exception, nonIntTensor.checkValid())

    def test_isTensor(self: TestRepShape) -> NoReturn:
        # NOTE: vscode gets confused and thinks this is unreachable code, but it's not
        self.assertTrue(RepShape((3, 4, 5)).isTensor())
        self.assertFalse(RepShape(3).isTensor())

    def test_isVector(self: TestRepShape) -> NoReturn:
        # NOTE: vscode gets confused and thinks this is unreachable code, but it's not
        self.assertTrue(RepShape(3).isVector())
        self.assertFalse(RepShape((3, 4, 5)).isVector())

    def test_width(self: TestRepShape) -> NoReturn:
        # NOTE: vscode gets confused and thinks this is unreachable code, but it's not
        self.assertEqual(RepShape((3, 4, 5)).width(), 5)
        self.assertNotEqual(RepShape((3, 4, 5)).width(), 4)
        self.assertEqual(RepShape(3).width(), 3)

    def test_height_and_depth(self: TestRepShape) -> NoReturn:
        # NOTE: vscode gets confused and thinks this is unreachable code, but it's not
        self.assertEqual(RepShape((3, 4, 5)).height(), 4)
        self.assertEqual(RepShape((3, 4, 5)).depth(), 3)
        self.assertNotEqual(RepShape((3, 4, 5)).height(), 5)
        self.assertNotEqual(RepShape((3, 4, 5)).depth(), 4)
        self.assertRaises(Exception, RepShape(3).height)
        self.assertRaises(Exception, RepShape(3).depth)

    def test_num_classes(self: TestRepShape) -> NoReturn:
        # NOTE this test will allow us to not forget to update things if we change
        # the functionality (i.e. it's a reminder, though I realize we should have a centralized
        # place to store these parameters, and I will soon...)
        self.assertEqual(RepShape.NUM_CLASSES, 10)

    # TODO test label to rep shape
    def test_label2RepShape(self: TestRepShape) -> NoReturn:
        # NOTE: vscode gets confused and thinks this is unreachable code, but it's not
        resnet: List[int] = [1, 2, 3, 4]
        label0: LayerLabel = LayerLabel(LayerLabel.INPUT, resnet)
        label1: LayerLabel = LayerLabel(LayerLabel.CONV1, resnet)
        label2: LayerLabel = LayerLabel((1, 0), resnet)
        label3: LayerLabel = LayerLabel((2, 1), resnet)
        label4: LayerLabel = LayerLabel((3, 2), resnet)
        label5: LayerLabel = LayerLabel((4, 3), resnet)
        label6: LayerLabel = LayerLabel(LayerLabel.FC, resnet)

        self.assertEqual(RepShape.label2outputRepShape(
            label0), RepShape((3, 32, 32)))
        self.assertEqual(RepShape.label2outputRepShape(
            label1), RepShape((64, 32, 32)))
        self.assertEqual(RepShape.label2outputRepShape(
            label2), RepShape((64, 32, 32)))
        self.assertEqual(RepShape.label2outputRepShape(
            label3), RepShape((128, 16, 16)))
        self.assertEqual(RepShape.label2outputRepShape(
            label4), RepShape((256, 8, 8)))
        self.assertEqual(RepShape.label2outputRepShape(
            label5), RepShape((512, 4, 4)))
        self.assertEqual(RepShape.label2outputRepShape(label6), RepShape(10))

    def test_stitchShapes(self: TestRepShape) -> NoReturn:
        # NOTE: vscode gets confused and thinks this is unreachable code, but it's not
        r1: List[int] = [1, 2, 1, 1]
        r2: List[int] = [1, 2, 2, 1]
        r1_shapes: List[RepShape] = [
            # Conv ->
            RepShape((64, 32, 32)),
            # (1, 0) ->
            RepShape((64, 32, 32)),
            # (2, 0) ->
            RepShape((128, 16, 16)),
            # (2, 1) ->
            RepShape((128, 16, 16)),
            # (3, 0) ->
            RepShape((256, 8, 8)),
            # (4, 0) ->
            RepShape((512, 4, 4)),
        ]
        r2_shapes: List[RepShape] = [
            # -> Conv
            RepShape((3, 32, 32)),
            # -> (1, 0)
            RepShape((64, 32, 32)),
            # -> (2, 0)
            RepShape((64, 32, 32)),
            # -> (2, 1)
            RepShape((128, 16, 16)),
            # -> (3, 0)
            RepShape((128, 16, 16)),
            # -> (3, 1)
            RepShape((256, 8, 8)),
            # -> (4, 0)
            RepShape((256, 8, 8)),
            # -> FC
            RepShape((512, 4, 4)),
        ]

        expected_shapes: List[List[RepShape]] = [
            [(r1_shapes[r1_idx], r2_shapes[r2_idx])
             for r2_idx in range(len(r2_shapes))]
            for r1_idx in range(len(r1_shapes))
        ]

        stitch_labels, _ = LayerLabel.generateTable(
            lambda l1, l2: (l1, l2), r1, r2)
        for i in range(len(stitch_labels)):
            for j in range(len(stitch_labels)):
                (label1, label2) = stitch_labels[i][j]
                self.assertEqual(RepShape.stitchShapes(
                    label1, label2), expected_shapes[i][j])


if __name__ == '__main__':
    unittest.main(verbosity=2)
