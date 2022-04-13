import torch.nn as nn

from typing import NoReturn, Type, Any, Callable, Union, List, Optional, Tuple


class StitchLabel(object):
    """
    A StitchLabel is basically either
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
    """
    INPUT = "input"
    OUTPUT = "output"
    CONV1 = "conv1"
    FC = "fc"

    def __init__(self, info: Union[Tuple[int, int], str]):
        self.str_label = None
        self.blockset = None
        self.block = None

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
                f"Cannot have type {type(info)} as a StitchLabel's internals")

    def checkValid(self) -> NoReturn:
        valid = not (self.str_label is None) or not (
            self.blockset is None or self.block is None)
        if not valid:
            raise Exception()

    @checkValid
    def isInput(self) -> bool:
        return self.str_label == StitchLabel.INPUT

    @checkValid
    def isOutput(self) -> bool:
        return self.str_label == StitchLabel.OUTPUT


class StitchGenerator(object):
    pass
