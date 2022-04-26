
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
    
    @staticmethod
    def transposed(table: List[List[T]]) -> List[List[T]]:
        return [[table[j][i] for  j in range(len(table))] for i in range(len(table[0]))]

class TableTester(unittest.TestCase):
    def test_mappedTable(self):
        table = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        mapped_should_be = [
            [1, 4, 9],
            [16, 25, 36],
            [49, 64, 81],
        ]

        mapped_is = Table.mappedTable(lambda x: x * x, table)

        self.assertEqual(mapped_is, mapped_should_be)
    
    def test_transposed(self):
        table = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        transposed_should_be = [
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9],
        ]

        transposed_is = Table.transposed(table)

        self.assertEqual(transposed_is, transposed_should_be)

if __name__ == "__main__":
    unittest.main(verbosity=2)
