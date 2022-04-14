# Enables type annotations using enclosing classes
from __future__ import annotations

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
    Tuple,
    TypeVar,
)

# https://docs.python.org/3/library/typing.html#typing.ParamSpec
T = TypeVar('T')
P = ParamSpec('P')


class RepShape(object):
    """
    RepShape is a simple wrapper for (int, int, int) | int, where it describes
    the shape of a representation. A representation for us is either a tensor
    with (height, width, dimensions) or simply a vector with a single dimension.
    We ignore the batch dimension.
    """

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
    def width(self: RepShape) -> int:
        # Width is meant to be overloaded to work with both types of reps
        return self.shape[1] if self.isTensor() else self.shape

    @checkValid
    def height(self: RepShape) -> int:
        assert self.isTensor()
        return self.shape[0]

    @checkValid
    def depth(self: RepShape) -> int:
        assert self.isTensor()
        return self.shape[2]

    @checkValid
    def isVector(self: RepShape):
        return type(self.shape) == int

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
        self.assertEqual(RepShape((3, 4, 5)).width(), 4)
        self.assertNotEqual(RepShape((3, 4, 5)).width(), 5)
        self.assertEqual(RepShape(3).width(), 3)

    def test_height_and_depth(self: TestRepShape) -> NoReturn:
        # NOTE: vscode gets confused and thinks this is unreachable code, but it's not
        self.assertEqual(RepShape((3, 4, 5)).height(), 3)
        self.assertEqual(RepShape((3, 4, 5)).depth(), 5)
        self.assertNotEqual(RepShape((3, 4, 5)).height(), 4)
        self.assertNotEqual(RepShape((3, 4, 5)).depth(), 4)
        self.assertRaises(Exception, RepShape(3).height)
        self.assertRaises(Exception, RepShape(3).depth)


if __name__ == '__main__':
    unittest.main(verbosity=2)
