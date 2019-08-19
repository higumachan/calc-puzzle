from enum import Enum
from typing import NamedTuple, List, Tuple
import math


class Operator(Enum):
    add = 0
    mul = 1

    def encode(self, x):
        if self == Operator.mul:
            return math.log2(x)
        return x

    def decode(self, x):
        if self == Operator.mul:
            return round(2 ** x)
        return x

class Block(NamedTuple):
    sum_number: int
    positions: List[Tuple[int, int]]
    operator: Operator


class Problem(NamedTuple):
    size: int
    blocks: List[Block]
