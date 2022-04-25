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

    """ 4 blocksets indexed by 1, while blocks are indexed by 0 """
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
            assert len(info) == 2, \
                "You can only have length (blockset: int, block: int) label tuples"
            assert type(info[0]) == int, \
                f"blockset was not int in label tuple, was {info[0]} of type {type(info[0])}"
            assert type(info[1]) == int, \
                f"block was not int in label tuple, was {info[1]} of type {type(info[1])}"
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
                # blockset/block must be integers
                if blockset != int(blockset) or block != int(block):
                    raise Exception(
                        f"LayerLabel representation ({self.blockset}, {self.block}) is of floats (must be ints)")

            return func(self, *args, **kwargs)
        return wrapper
    ################################################################################

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
    ################################################################################

    @checkValid
    def isInput(self: LayerLabel) -> bool:
        return self.str_label == LayerLabel.INPUT

    @checkValid
    def isConv1(self: LayerLabel) -> bool:
        return self.str_label == LayerLabel.CONV1

    @checkValid
    def isBlock(self: LayerLabel) -> bool:
        return self.str_label is None

    @checkValid
    def isFc(self: LayerLabel) -> bool:
        return self.str_label == LayerLabel.FC

    @checkValid
    def isOutput(self: LayerLabel) -> bool:
        return self.str_label == LayerLabel.OUTPUT
    ################################################################################

    @checkValid
    def getBlockset(self: LayerLabel) -> int:
        assert self.isBlock(), "Cannot get blockset of a string label"
        return self.blockset

    @checkValid
    def getBlock(self: LayerLabel) -> int:
        assert self.isBlock(), "Cannot get block of a string label"
        return self.block

    ################################################################################

    @checkValid
    def prevLabel(self: LayerLabel) -> LayerLabel:
        # Get the label that would represent the layer before the one repped by this label
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
        elif str_label == LayerLabel.FC:
            blockset = 4
            block = model_blocksets[3] - 1
            return LayerLabel((blockset, block), model_blocksets)
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
    def __sub__(self: LayerLabel, other: int) -> LayerLabel:
        assert type(other) == int, "other must be an int in __sub__"
        assert other >= 0, "other must be non-negative in __sub__"
        label: LayerLabel = self
        for _ in range(other):
            label = label.prevLabel()
        return label
    ################################################################################

    @checkValid
    def label2idx(self: LayerLabel) -> int:
        # Return the index of the layer in the model assuming that
        # the model is input -> conv1 -> blocksets -> fc -> output
        # 0-indexing by conv1. NOTE that blocks are treated as single
        # layers. Used to index into tables.
        if self.str_label == LayerLabel.INPUT:
            raise Exception("Input has no layer index")
        elif self.str_label == LayerLabel.CONV1:
            return 0
        elif self.str_label is None:
            blockset: int = self.blockset
            block: int = self.block
            assert not blockset is None, "blockset cannot be None if str_label is None (label2idx)"
            assert not block is None, "block cannot be None if str_label is None (label2idx)"

            block_idx: int = block - LayerLabel.BLOCK_MIN
            blockset_idx: int = blockset - LayerLabel.BLOCKSET_MIN
            prefix: List[int] = self.model_blocksets[:blockset_idx]
            prefix_num_blocks: int = sum(prefix)
            # conv1 + previous blocks
            return 1 + prefix_num_blocks + block_idx

        elif self.str_label == LayerLabel.FC:
            # conv1 + blocksets (number of blocks per blockset in the list) + fc
            num_layers: int = 2 + sum(self.model_blocksets)
            return num_layers - 1
        else:
            raise Exception(
                f"Not Supported: label2idx({self.str_label} | ({self.blockset}, {self.block}))")
    ################################################################################

    @staticmethod
    def numLayers(R: List[int]) -> int:
        return 2 + sum(R)

    @staticmethod
    def idx2label(idx: int, R: List[int]) -> LayerLabel:
        # Given an index, return the label that represents it
        if idx == 0:
            return LayerLabel(LayerLabel.CONV1, R)
        else:
            num_layers: int = LayerLabel.numLayers(R)
            fc_idx: int = num_layers - 1
            if idx == fc_idx:
                return LayerLabel(LayerLabel.FC, R)
            elif idx < fc_idx:
                remaining_idx: int = idx - 1
                blockset_idx = 0
                while remaining_idx >= R[blockset_idx]:
                    remaining_idx -= R[blockset_idx]
                    blockset_idx += 1

                blockset: int = blockset_idx + LayerLabel.BLOCKSET_MIN
                block: int = remaining_idx + LayerLabel.BLOCK_MIN
                return LayerLabel((blockset, block), R)
            else:
                raise Exception(
                    f"idx {idx} is not a valid index in LayerLabel.idx2label")

    @staticmethod
    def labels(R: List[int]) -> List[LayerLabel]:
        # Get a list of labels for a given ResNet
        l = []
        l.append(LayerLabel(LayerLabel.CONV1, R))
        for blockset, num_blocks in enumerate(R, LayerLabel.BLOCKSET_MIN):
            for block in range(LayerLabel.BLOCK_MIN, LayerLabel.BLOCK_MIN + num_blocks):
                l.append(LayerLabel((blockset, block), R))
        l.append(LayerLabel(LayerLabel.FC, R))
        return l

    @staticmethod
    def generateTable(constructor: Callable[[LayerLabel, LayerLabel], T], R1: List[int], R2: List[int]) -> Tuple[List[List[T]], Dict[(int, int), Tuple[LayerLabel, LayerLabel]]]:
        # Generate a table of objects of a given type for two networks. The use cases are:
        # - stitches: the constructor is a stitch constructor
        # - stitched networks: the constructor is a stitched network constructor
        # - similarities: the constructor is a training function on a pre-existing table/networks
        # - mean2 differences: the constructor is a mean2 difference function on a pre-existing table/networks
        # NOTE: we stitch from all layers except FC, but we stitch INTO all layers
        num_layers1: int = LayerLabel.numLayers(R1)
        num_layers2: int = LayerLabel.numLayers(R2)

        height = num_layers1 - 1
        width = num_layers2 - 2

        # all of R1, except FC sends INTO all of R2
        labels1: List[LayerLabel] = LayerLabel.labels(R1)
        labels1.pop()

        labels2: List[LayerLabel] = LayerLabel.labels(R2)
        # NOTE his is a quick hack to be able to avoid -> conv1 for the old resnet
        # avoid sending into conv1 because old resnet does not support
        # avoid sending into fc because old resnet does it differently
        labels2 = labels2[1:-1]
        assert len(labels1) == height, f"Expected height {height}, but got {len(labels1)}"
        assert len(labels2) == width, f"Expected length {width}, but got {len(labels2)}"
        idx2label = {
            (i, j): (labels1[i], labels2[j]) for i in range(height) for j in range(width)
        }
        return [[constructor(*idx2label[(i, j)]) for j in range(width)] for i in range(height)], idx2label


class TestLayerLabel(unittest.TestCase):
    """
    Extensive unit testing for LayerLabel. This mainly checks that
    - checkValid is being used to confirm the correctness of the representation properly
    - equality works (which should imply that the representation, hashing, etc... are OK)
    - you can get previous labels given labels (used to know which layers are compared when
      you have the label of the sender layer and the label of the reciever layer)
    - conversion to and from layers and indices works
    - you are able to get an array that maps indices to labels for a single network (i.e. it
      orders the layers by index)
    - you can generate tables (just doubly nested Arrays), given a closure, to do things like,
      for example, create stitch tables (and this also returns a mapping from index tuples to
      label tuples representing the sending and recieving layers in their corresponding networks)
    """
    R2222: List[int] = [2, 2, 2, 2]
    R1111: List[int] = [1, 1, 1, 1]
    R1234: List[int] = [1, 2, 3, 4]

    def test_checkValid(self: TestLayerLabel) -> NoReturn:
        """ Check that an exception is thrown when the representation of this label is invalid """
        invalidStr: str = "invalidString"
        invalidTuple2: Tuple[int, int] = (-1, -1)
        invalidTuple3: Tuple[int, int] = (1, 1)
        invalidTuple4: Tuple[int, int] = (1.1, 1.2)
        input1 = LayerLabel(invalidStr, TestLayerLabel.R1111)
        input2 = LayerLabel(invalidTuple2, TestLayerLabel.R1111)
        input3 = LayerLabel(invalidTuple3, TestLayerLabel.R1111)
        input4 = LayerLabel((1, 1), TestLayerLabel.R2222)
        input4.block = 2.2
        input4.blockset = 1.1
        # NOTE: when you do assertRaises you need to provide a Callable
        self.assertRaises(Exception, input1.checkValid())
        self.assertRaises(Exception, input2.checkValid())
        self.assertRaises(Exception, input3.checkValid())
        self.assertRaises(Exception, input4.checkValid())
        self.assertRaises(Exception,
                          lambda: LayerLabel((1.1, 2.2), TestLayerLabel.R2222))

    def test_equals(self: TestLayerLabel) -> NoReturn:
        """ Check that equality works """
        b1: LayerLabel = LayerLabel((1, 0), TestLayerLabel.R2222)
        b2: LayerLabel = LayerLabel((1, 0), TestLayerLabel.R2222)
        b3: LayerLabel = LayerLabel((2, 0), TestLayerLabel.R2222)
        b4: LayerLabel = LayerLabel((1, 1), TestLayerLabel.R2222)
        s1: LayerLabel = LayerLabel(LayerLabel.CONV1, TestLayerLabel.R2222)
        s2: LayerLabel = LayerLabel(LayerLabel.CONV1, TestLayerLabel.R2222)
        s3: LayerLabel = LayerLabel(LayerLabel.FC, TestLayerLabel.R2222)
        s4: LayerLabel = LayerLabel(LayerLabel.CONV1, TestLayerLabel.R1234)

        # Make sure that identical objects (underlying: tuple) are equal
        self.assertTrue(b1 == b1)
        self.assertTrue(b1 == b2)
        self.assertTrue(b2 == b1)

        # Make sure that strings and non-identical objects are not equal
        self.assertFalse(b1 == s1)
        self.assertFalse(s1 == b1)
        self.assertFalse(b1 == s2)

        # Make sure that identical objects (underlying: string) are equal
        self.assertTrue(s1 == s1)
        self.assertTrue(s1 == s2)
        self.assertTrue(s2 == s1)

        # Make sure that if you vary the blockset, you get a different object
        self.assertFalse(s1 == s4)

        # Make sure that if you vary the tuple you get a different object
        self.assertFalse(b1 == b3)
        self.assertFalse(b1 == b4)

        # Make sure that if you vary the string you get a different object
        self.assertFalse(s1 == s3)

    def test_prevLabel(self: TestLayerLabel) -> NoReturn:
        """  Check that lbl.prevLabel() and shorthand `lbl - 1` works with cached model blocksets"""
        input1: LayerLabel = LayerLabel((1, 1), TestLayerLabel.R2222)
        expected1: LayerLabel = LayerLabel((1, 0), TestLayerLabel.R2222)
        expected2: LayerLabel = LayerLabel(
            LayerLabel.CONV1, TestLayerLabel.R2222)

        input2: LayerLabel = LayerLabel((3, 0), TestLayerLabel.R1234)
        expected3: LayerLabel = LayerLabel((2, 1), TestLayerLabel.R1234)
        expected4: LayerLabel = LayerLabel((2, 0), TestLayerLabel.R1234)
        expected5: LayerLabel = LayerLabel((1, 0), TestLayerLabel.R1234)
        expected6: LayerLabel = LayerLabel(
            LayerLabel.CONV1, TestLayerLabel.R1234)

        # Check that the prevLabel() method works and that __sub__ calls it properly
        self.assertEqual(input1.prevLabel(), expected1)
        self.assertEqual(input1 - 1, expected1)
        self.assertEqual(input1.prevLabel().prevLabel(), expected2)
        self.assertEqual(input1 - 2, expected2)
        self.assertEqual(input1 - 1 - 1, expected2)

        # Check that when size is changing, it still works
        self.assertEqual(input2 - 1, expected3)
        self.assertEqual(input2 - 2, expected4)
        self.assertEqual(input2 - 3, expected5)
        self.assertEqual(input2 - 4, expected6)

    def test_label2idx_and_idx2layer(self: TestLayerLabel) -> NoReturn:
        """ Check that lbl.label2idx() works properly """
        # [0: conv1, 1: (1, 0), 2: (1, 1), ...]
        label1: LayerLabel = LayerLabel((1, 1), TestLayerLabel.R2222)
        idx1: int = 2

        # [0: conv1, 1: (1, 0), 2: (2, 0), ...]
        label2: LayerLabel = LayerLabel(LayerLabel.CONV1, TestLayerLabel.R1111)
        label3: LayerLabel = LayerLabel((1, 0), TestLayerLabel.R1111)
        label4: LayerLabel = LayerLabel((2, 0), TestLayerLabel.R1111)
        idx2: int = 0
        idx3: int = 1
        idx4: int = 2

        label5: LayerLabel = LayerLabel(LayerLabel.CONV1, TestLayerLabel.R1234)
        label6: LayerLabel = LayerLabel((1, 0), TestLayerLabel.R1234)
        label7: LayerLabel = LayerLabel((2, 0), TestLayerLabel.R1234)
        label8: LayerLabel = LayerLabel((2, 1), TestLayerLabel.R1234)
        label9: LayerLabel = LayerLabel((3, 0), TestLayerLabel.R1234)
        label10: LayerLabel = LayerLabel((3, 1), TestLayerLabel.R1234)
        label11: LayerLabel = LayerLabel((3, 2), TestLayerLabel.R1234)
        idx5: int = 0
        idx6: int = 1
        idx7: int = 2
        idx8: int = 3
        idx9: int = 4
        idx10: int = 5
        idx11: int = 6

        # Check that label2idx() works
        self.assertEqual(label1.label2idx(), idx1)
        self.assertEqual(label2.label2idx(), idx2)
        self.assertEqual(label3.label2idx(), idx3)
        self.assertEqual(label4.label2idx(), idx4)
        self.assertEqual(label5.label2idx(), idx5)
        self.assertEqual(label6.label2idx(), idx6)
        self.assertEqual(label7.label2idx(), idx7)
        self.assertEqual(label8.label2idx(), idx8)
        self.assertEqual(label9.label2idx(), idx9)
        self.assertEqual(label10.label2idx(), idx10)
        self.assertEqual(label11.label2idx(), idx11)

        # Check that idx2label() works
        self.assertEqual(LayerLabel.idx2label(
            idx1, TestLayerLabel.R2222), label1)
        self.assertEqual(LayerLabel.idx2label(
            idx2, TestLayerLabel.R1111), label2)
        self.assertEqual(LayerLabel.idx2label(
            idx3, TestLayerLabel.R1111), label3)
        self.assertEqual(LayerLabel.idx2label(
            idx4, TestLayerLabel.R1111), label4)
        self.assertEqual(LayerLabel.idx2label(
            idx5, TestLayerLabel.R1234), label5)
        self.assertEqual(LayerLabel.idx2label(
            idx6, TestLayerLabel.R1234), label6)
        self.assertEqual(LayerLabel.idx2label(
            idx7, TestLayerLabel.R1234), label7)
        self.assertEqual(LayerLabel.idx2label(
            idx8, TestLayerLabel.R1234), label8)
        self.assertEqual(LayerLabel.idx2label(
            idx9, TestLayerLabel.R1234), label9)
        self.assertEqual(LayerLabel.idx2label(
            idx10, TestLayerLabel.R1234), label10)
        self.assertEqual(LayerLabel.idx2label(
            idx11, TestLayerLabel.R1234), label11)

    def test_labels(self: TestLayerLabel) -> NoReturn:
        """ Check that if you get the labels for a layer of a network it is correct """
        labels1: List[LayerLabel] = LayerLabel.labels(TestLayerLabel.R1111)
        labels2: List[LayerLabel] = LayerLabel.labels(TestLayerLabel.R2222)
        labels3: List[LayerLabel] = LayerLabel.labels(TestLayerLabel.R1234)

        expected1: List[LayerLabel] = [
            LayerLabel(LayerLabel.CONV1, TestLayerLabel.R1111),
            LayerLabel((1, 0), TestLayerLabel.R1111),
            LayerLabel((2, 0), TestLayerLabel.R1111),
            LayerLabel((3, 0), TestLayerLabel.R1111),
            LayerLabel((4, 0), TestLayerLabel.R1111),
            LayerLabel(LayerLabel.FC, TestLayerLabel.R1111),
        ]
        expected2: List[LayerLabel] = [
            LayerLabel(LayerLabel.CONV1, TestLayerLabel.R2222),
            LayerLabel((1, 0), TestLayerLabel.R2222),
            LayerLabel((1, 1), TestLayerLabel.R2222),
            LayerLabel((2, 0), TestLayerLabel.R2222),
            LayerLabel((2, 1), TestLayerLabel.R2222),
            LayerLabel((3, 0), TestLayerLabel.R2222),
            LayerLabel((3, 1), TestLayerLabel.R2222),
            LayerLabel((4, 0), TestLayerLabel.R2222),
            LayerLabel((4, 1), TestLayerLabel.R2222),
            LayerLabel(LayerLabel.FC, TestLayerLabel.R2222),
        ]
        expected3: List[LayerLabel] = [
            LayerLabel(LayerLabel.CONV1, TestLayerLabel.R1234),
            LayerLabel((1, 0), TestLayerLabel.R1234),
            LayerLabel((2, 0), TestLayerLabel.R1234),
            LayerLabel((2, 1), TestLayerLabel.R1234),
            LayerLabel((3, 0), TestLayerLabel.R1234),
            LayerLabel((3, 1), TestLayerLabel.R1234),
            LayerLabel((3, 2), TestLayerLabel.R1234),
            LayerLabel((4, 0), TestLayerLabel.R1234),
            LayerLabel((4, 1), TestLayerLabel.R1234),
            LayerLabel((4, 2), TestLayerLabel.R1234),
            LayerLabel((4, 3), TestLayerLabel.R1234),
            LayerLabel(LayerLabel.FC, TestLayerLabel.R1234),
        ]

        self.assertEqual(labels1, expected1)
        self.assertEqual(labels2, expected2)
        self.assertEqual(labels3, expected3)

    def test_net_size(self: TestLayerLabel) -> NoReturn:
        input1: List[int] = TestLayerLabel.R1111
        input2: List[int] = TestLayerLabel.R2222
        input3: List[int] = TestLayerLabel.R1234
        expected1: int = 6
        expected2: int = 10
        expected3: int = 12

        self.assertEqual(LayerLabel.numLayers(input1), expected1)
        self.assertEqual(LayerLabel.numLayers(input2), expected2)
        self.assertEqual(LayerLabel.numLayers(input3), expected3)

    def test_generateTable(self: TestLayerLabel) -> NoReturn:
        """ Check that you can generate a table given a constructor closure"""
        constructor: Callable[[LayerLabel, LayerLabel], Tuple[int, int]] = \
            lambda l1, l2: (l1.label2idx(), l2.label2idx())

        netpairs: List[Tuple[List[int], List[int]]] = [
            (TestLayerLabel.R1111, TestLayerLabel.R1111),
            (TestLayerLabel.R1111, TestLayerLabel.R2222),
            (TestLayerLabel.R1111, TestLayerLabel.R1234),
            (TestLayerLabel.R2222, TestLayerLabel.R2222),
            (TestLayerLabel.R2222, TestLayerLabel.R1234),
            (TestLayerLabel.R1234, TestLayerLabel.R1234),
        ]

        # All layers of R1 (except FC) stitch into all layers of R12
        # So indices 0, 1, 2, ... through to the last (not inclusive) all go into
        # indices 0, 1, 2, ... through to the last (inclusive).
        tables: List[List[Tuple[int, int]]] = [
            LayerLabel.generateTable(constructor, r1, r2) for (r1, r2) in netpairs]
        expectedTablesAndIdxs2Labels: List[List[Tuple[int, int]]] = [
            (
                # The table is generated by the constructor closure, which in our case
                # simply turns labels into idxs.
                [[(i, j) \
                    for j in range(LayerLabel.numLayers(r2))] for i in range(LayerLabel.numLayers(r1) - 1)],
                # The idx2labels is generated automatically to always look like this
                {(i, j): (LayerLabel.idx2label(i, r1), LayerLabel.idx2label(j, r2)) \
                    for i in range(LayerLabel.numLayers(r1) - 1) for j in range(LayerLabel.numLayers(r2))}
            )
            for (r1, r2) in netpairs
        ]

        self.assertEqual(len(tables), len(netpairs))
        self.assertEqual(len(tables), len(expectedTablesAndIdxs2Labels))
        self.assertEqual(tables, expectedTablesAndIdxs2Labels)


if __name__ == '__main__':
    unittest.main(verbosity=2)
