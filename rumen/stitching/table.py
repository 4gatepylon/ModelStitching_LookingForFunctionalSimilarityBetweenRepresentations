
# Enables type annotations using enclosing classes
from __future__ import annotations

import unittest

from typing import (
    Callable,
    List,
    TypeVar,
)


T = TypeVar('T')
G = TypeVar('G')

# TODO refactor to use https://numpy.org/doc/stable/reference/arrays.html
# numpy arrays of object types

# NOTE that ideally we'd like to make this lazy, so we can keep calling mapped table
# without actually having to store all the intermediate results in memory.


class Table(object):
    # TODO it would be super helpful to have zipped tables and other utility

    @staticmethod
    def mappedTable(transform: Callable[[T], G], table: List[List[T]]) -> List[List[G]]:
        return [[transform(x) for x in row] for row in table]
