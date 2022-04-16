
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


class Table(object):
    @staticmethod
    def mappedTable(table: List[List[T]], transform: Callable[[T], G]) -> List[List[G]]:
        return [[transform(x) for x in row] for row in table]
