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

# https://docs.python.org/3/library/typing.html#typing.ParamSpec
T = TypeVar('T')
P = ParamSpec('P')


class StitchedResnet(nn.Module):
    def __init__(self, net1, net2, snd_label, rcv_label, stitch):
        super(StitchedResnet, self).__init__()
        self.sender = net1
        self.reciever = net2
        self.snd_lbl = snd_label
        self.rcv_lbl = rcv_label
        self.stitch = stitch

    def freeze_sender(self: StitchedResnet) -> NoReturn:
        raise NotImplementedError

    def freeze_reciever(self: StitchedResnet) -> NoReturn:
        raise NotImplementedError

    def freeze(self: StitchedResnet) -> NoReturn:
        raise NotImplementedError

    def forward(self, x):
        h = self.sender(x, vent=self.snd_lbl, into=False)
        h = self.stitch(h)
        h = self.reciever(h, vent=self.rcv_lbl, into=True)
        return h


class TestStitchedResnet(unittest.TestCase):
    def test_nothing(self: TestStitchedResnet) -> NoReturn:
        raise NotImplementedError


if __name__ == '__main__':
    unittest.main(verbosity=2)
