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
    List,
    Optional,
    Tuple,
    TypeVar,
)
# https://docs.python.org/3/library/typing.html#typing.ParamSpec
T = TypeVar('T')
P = ParamSpec('P')


class LayerLabel(object):
    """
    A LayerLabel for a sepcific network is basically either
    a tuple of integers representing (blockset, block)
    or a string representing a specific layer outside the
    center four blocksets. NOTE: this is meant specifically
    for ResNets that have 4 blocksets each with an arbitrary
    number of blocks within them and potentially other
    layers before and after, labeled using special labels.

    By convention the other blocks are either `conv1`, `fc`, 
    `input` or `output` (the latter two of which are special
    reserved words with the property that you can NEVER
    stitch out of output and you can never stitch INTO input).

    NOTE that blockset is 1-indexed, while block is 0-indexed.
    """
    INPUT: str = "input"
    OUTPUT: str = "output"
    CONV1: str = "conv1"
    FC: str = "fc"

    """ Documentation, basically """
    BLOCKSET_MIN: int = 1
    BLOCK_MIN: int = 0
    BLOCKSET_MAX: int = 4

    def __init__(self: LayerLabel, info: Union[Tuple[int, int], str], model_blocksets: List[int]):
        self.str_label: Optional[str] = None
        self.blockset: Optional[str] = None
        self.block: Optional[str] = None
        self.model_blocksets: List[int] = model_blocksets

        if type(info) == str:
            self.str_label = info
        elif type(info) == tuple and len(info) == 2:
            assert len(
                info) == 2, "You can only have length (blockset: int, block: int) label tuples"
            assert type(info[0]) == int, "blockset was not int in label tuple"
            assert type(info[1]) == int, "block was not int in label tuple"
            self.blockset = info[0]
            self.block = info[1]
        else:
            raise Exception(
                f"Cannot have type {type(info)} as a LayerLabel's internals")

    def checkValid(func: Callable[Concatenate[LayerLabel, P], T]) -> Callable[Concatenate[LayerLabel, P], T]:
        # Checks that the representation of this label is valid
        def wrapper(self: LayerLabel, *args: P.args, **kwargs: P.kwargs):
            # Must either be string label or a tuple of (blockset, block)
            valid: bool = (not (self.str_label is None)) or \
                ((not (self.blockset is None or self.block is None))
                 and self.blockset >= 0 and self.block >= 0)
            if not valid:
                raise Exception(
                    f"LayerLabel representation {self.str_label} | ({self.blockset}, {self.block}) is not valid")
            if not self.str_label is None:
                # If it is a string it must be a supported one
                supported: bool = self.str_label in [
                    LayerLabel.INPUT,
                    LayerLabel.CONV1,
                    LayerLabel.FC,
                    LayerLabel.OUTPUT
                ]
                if not supported:
                    raise Exception(
                        f"LayerLabel {self.str_label} is not supported")
            else:
                # If it is not a string, but instead a tuple, it must be within the bounds of the ResNet
                blockset = self.blockset
                block = self.block
                if blockset < LayerLabel.BLOCKSET_MIN or blockset > LayerLabel.BLOCKSET_MAX:
                    raise Exception(
                        f"Blockset must be in [{LayerLabel.BLOCKSET_MIN}, {LayerLabel.BLOCKSET_MAX}]")
                maxBlock = self.model_blocksets[blockset - 1] - 1 + \
                    LayerLabel.BLOCK_MIN
                if block < LayerLabel.BLOCK_MIN or block > maxBlock:
                    raise Exception(
                        f"Block must be in [{LayerLabel.BLOCK_MIN}, {maxBlock}], was {block}")

            return func(self, *args, **kwargs)
        return wrapper

    @checkValid
    def isInput(self: LayerLabel) -> bool:
        return self.str_label == LayerLabel.INPUT

    @checkValid
    def isOutput(self: LayerLabel) -> bool:
        return self.str_label == LayerLabel.OUTPUT

    @checkValid
    def prevLabel(self: LayerLabel) -> LayerLabel:
        model_blocksets = self.model_blocksets
        blockset = self.blockset
        block = self.block
        str_label = self.str_label

        assert not model_blocksets is None, "model_blocksets was None"
        assert type(model_blocksets) == list and len(model_blocksets) == 4,\
            "model_blocksets must be a list of 4 blocksets"
        assert min(model_blocksets) >= 0, "model_blocksets must be non-negative"
        if str_label == LayerLabel.INPUT:
            raise Exception("There is no prevLabel(LayerLabel.INPUT)")
        elif str_label == LayerLabel.CONV1:
            return LayerLabel(LayerLabel.INPUT, model_blocksets)
        elif str_label is None:
            assert not blockset is None, "blockset cannot be None if str_label is None"
            assert not block is None, "block cannot be None if str_label is None"
            blockset_idx: int = blockset - LayerLabel.BLOCKSET_MIN
            block_idx: int = block - LayerLabel.BLOCK_MIN
            if block_idx > 0:
                # If you can stay in the same blockset, stay
                return LayerLabel((blockset, block - 1), model_blocksets)
            elif blockset_idx > 0:
                # If you cannot stay in the same blockset, try to move to the previous
                prev_blockset_num_blocks: int = model_blocksets[blockset_idx - 1]
                prev_blockset_max_block: int = prev_blockset_num_blocks - 1
                return LayerLabel((blockset - 1,  prev_blockset_max_block), model_blocksets)
            else:
                # If there is no previous blockset, you become CONV1
                return LayerLabel(LayerLabel.CONV1, model_blocksets)
        elif str_label == LayerLabel.OUTPUT:
            return LayerLabel(LayerLabel.FC, model_blocksets)
        else:
            raise Exception(
                f"Not Supported: prevLabel({str_label} | ({blockset}, {block}))")

    @checkValid
    def __repr__(self: LayerLabel) -> str:
        if self.str_label is None:
            return str((self.blockset, self.block))
        else:
            return self.str_label

    @checkValid
    def __str__(self: LayerLabel) -> str:
        return repr(self)

    @checkValid
    def __eq__(self: LayerLabel, other: LayerLabel) -> bool:
        return type(other) == LayerLabel and \
            self.str_label == other.str_label and \
            self.blockset == other.blockset and self.block == other.block and \
            self.model_blocksets == other.model_blocksets

    @checkValid
    def __hash__(self) -> int:
        return repr(self)

    @checkValid
    def __sub__(self: LayerLabel, other: int) -> LayerLabel:
        assert type(other) == int, "other must be an int"
        assert other >= 0, "other must be non-negative"
        label: LayerLabel = self
        for _ in range(other):
            label = label.prevLabel()
        return label

    @checkValid
    def layerIndex(self: LayerLabel) -> int:
        # Return the index of the layer in the model assuming that
        # the model is input -> conv1 -> blocksets -> fc -> output
        # 0-indexing by conv1. NOTE that blocks are treated as single
        # layers. Used to index into tables.
        if self.str_label == LayerLabel.INPUT:
            raise Exception("Input has no layer index")
        elif self.str_label == LayerLabel.CONV1:
            return 0
        elif self.str_label is None:
            blockset = self.blockset
            block = self.block
            assert not blockset is None, "blockset cannot be None if str_label is None (layerIndex)"
            assert not block is None, "block cannot be None if str_label is None (layerIndex)"

            block_idx = block - LayerLabel.BLOCK_MIN
            blockset_idx: int = blockset - LayerLabel.BLOCKSET_MIN
            prefix: List[int] = self.model_blocksets[:blockset_idx]
            prefix_num_blocks: int = sum(prefix)
            # conv1 + previous blocks
            return 1 + prefix_num_blocks + block_idx

        elif self.str_label == LayerLabel.FC:
            # conv1 + blocksets (number of blocks per blockset in the list) + fc
            num_layers = 2 + sum(self.model_blocksets)
            return num_layers - 1
        else:
            raise Exception(
                f"Not Supported: layerIndex({self.str_label} | ({self.blockset}, {self.block}))")


class TestLayerLabel(unittest.TestCase):
    R2222 = [2, 2, 2, 2]
    R1111 = [1, 1, 1, 1]

    def test_checkValid(self):
        """ Check that an exception is thrown when the representation of this label is invalid """
        invalidStr: str = "invalidString"
        invalidTuple2: Tuple[int, int] = (-1, -1)
        invalidTuple3: Tuple[int, int] = (1, 1)
        input1 = LayerLabel(invalidStr, TestLayerLabel.R1111)
        input2 = LayerLabel(invalidTuple2, TestLayerLabel.R1111)
        input3 = LayerLabel(invalidTuple3, TestLayerLabel.R1111)
        # NOTE: when you do assertRaises you need to provide a Callable
        self.assertRaises(Exception, input1.prevLabel)
        self.assertRaises(Exception, input2.prevLabel)
        self.assertRaises(Exception, input3.prevLabel)

    def test_equals(self):
        """ Check that equality works """
        b1: LayerLabel = LayerLabel((1, 0), TestLayerLabel.R1111)
        b2: LayerLabel = LayerLabel((1, 0), TestLayerLabel.R1111)
        s1: LayerLabel = LayerLabel("conv1", TestLayerLabel.R1111)
        s2: LayerLabel = LayerLabel("conv1", TestLayerLabel.R1111)

        self.assertTrue(b1 == b1)
        self.assertTrue(b1 == b2)
        self.assertTrue(b2 == b1)

        self.assertFalse(b1 == s1)
        self.assertFalse(s1 == b1)
        self.assertFalse(b1 == s2)

        self.assertTrue(s1 == s1)
        self.assertTrue(s1 == s2)
        self.assertTrue(s2 == s1)

    def test_prevLabel(self):
        """  Test that lbl.prevLabel() and shorthand `lbl - 1` works with cached model blocksets"""
        input: LayerLabel = LayerLabel((1, 1), TestLayerLabel.R2222)
        expected1: LayerLabel = LayerLabel((1, 0), TestLayerLabel.R2222)
        expected2: LayerLabel = LayerLabel("conv1", TestLayerLabel.R2222)

        self.assertEqual(input.prevLabel(), expected1)
        self.assertEqual(input - 1, expected1)
        self.assertEqual(input.prevLabel().prevLabel(), expected2)
        self.assertEqual(input - 2, expected2)
        self.assertEqual(input - 1 - 1, expected2)

    def test_layerIndex(self):
        """ Test that lbl.layerIndex() works properly """
        # [0: conv1, 1: (1, 0), 2: (1, 1), ...]
        input1: LayerLabel = LayerLabel((1, 1), TestLayerLabel.R2222)
        expected1: int = 2

        # [0: conv1, 1: (1, 0), 2: (2, 0), ...]
        input2 = LayerLabel(LayerLabel.CONV1, TestLayerLabel.R1111)
        input3: LayerLabel = LayerLabel((1, 0), TestLayerLabel.R1111)
        input4 = LayerLabel((2, 0), TestLayerLabel.R1111)

        expected2: int = 0
        expected3: int = 1
        expected4: int = 2

        self.assertEqual(input1.layerIndex(), expected1)
        self.assertEqual(input2.layerIndex(), expected2)
        self.assertEqual(input3.layerIndex(), expected3)
        self.assertEqual(input4.layerIndex(), expected4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
