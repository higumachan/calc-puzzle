from enum import Enum
from typing import NamedTuple, List, Tuple


class Operator(Enum):
    add = 0
    mul = 1

    def op(self, x, y):
        if self == Operator.add:
            return x + y
        if self == Operator.mul:
            return x * y

    def identity(self):
        if self == Operator.add:
            return 0
        if self == Operator.mul:
            return 1


class Block(NamedTuple):
    agg_number: int
    positions: List[Tuple[int, int]]
    operator: Operator


class Problem(NamedTuple):
    size: int
    blocks: List[Block]
