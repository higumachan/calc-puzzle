from typing import NamedTuple, List, Tuple


class Block(NamedTuple):
    sum_number: int
    positions: List[Tuple[int, int]]


class Problem(NamedTuple):
    size: int
    blocks: List[Block]
