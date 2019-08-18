from typing import Tuple, Optional

from calc_puzzle.structs import Problem, Block
import pulp
import numpy as np
import functools


def create_mip_problem(problem: Problem) -> Tuple[pulp.LpProblem, dict]:
    prob = pulp.LpProblem("PlusPuzzle", pulp.LpMinimize)
    prob += 0

    size = problem.size
    numbers = range(1, size + 1)  # [1, 6)
    xs = range(1, size + 1)
    ys = range(1, size + 1)

    choices = pulp.LpVariable.dicts("Cell", (numbers, xs, ys), 0, 1, pulp.LpInteger)

    # 1つのマスに入る値は1つだけ
    for x in xs:
        for y in ys:
            prob += pulp.lpSum([choices[v][x][y] for v in numbers]) == 1

    # 縦横で同じ数字は1つしか入らない
    for v in numbers:
        for y in ys:
            prob += pulp.lpSum([choices[v][x][y] for x in xs]) == 1

    for v in numbers:
        for x in xs:
            prob += pulp.lpSum([choices[v][x][y] for y in ys]) == 1

    # 問題のブロックの制約を追加

    for block in problem.blocks:
        prob += functools.reduce(
            block.operator.op,
            [v * choices[v][x][y] for v in numbers for x, y in block.positions],
            block.operator.identity()) == block.agg_number
        # prob += pulp.lpSum([v * choices[v][x][y] for v in numbers for x, y in block.positions]) == block.sum_number

    return prob, choices


def solve(problem: Problem) -> Optional[np.ndarray]:
    size = problem.size
    numbers = range(1, size + 1)  # [1, 6)
    xs = range(1, size + 1)
    ys = range(1, size + 1)
    prob, choices = create_mip_problem(problem)
    s = prob.solve()

    if s != 1:
        return None

    answer_board = np.zeros((size, size), dtype=np.int32)
    for y in ys:
        for x in xs:
            for v in numbers:
                if choices[v][x][y].value() == 1:
                    answer_board[y-1][x-1] = v
    return answer_board


def is_unique(problem: Problem):
    size = problem.size
    numbers = range(1, size + 1)  # [1, 6)
    xs = range(1, size + 1)
    ys = range(1, size + 1)

    prob, choices = create_mip_problem(problem)

    s = prob.solve()

    assert s == 1

    for y in ys:
        for x in xs:
            for v in numbers:
                if choices[v][x][y].value() == 1:
                    prob2, choices2 = create_mip_problem(problem)
                    prob2 += choices2[v][x][y] == 0
                    if prob2.solve() != -1:
                        return False
    return True


if __name__ == '__main__':
    problem = Problem(5, [
        Block(4, [(1, 1), (2, 1), (1, 2)]),
        Block(24, [(3, 1), (4, 1), (5, 1), (2, 2), (3, 2), (4, 2)]),
        Block(5, [(5, 2), (5, 3), (4, 3)]),
        Block(7, [(2, 3), (3, 3)]),
        Block(12, [(1, 3), (1, 4), (1, 5)]),
        Block(5, [(2, 4), (2, 5)]),
        Block(4, [(3, 4), (3, 5), (4, 5)]),
        Block(14, [(4, 4), (5, 4), (5, 5)]),
    ])

    print(solve(problem))

    print(is_unique(problem))

    not_unique_problem = Problem(5, [
        Block(sum(range(1, 6)) * 5, [(x, y) for y in range(1, 6) for x in range(1, 6)])
    ])

    print(solve(not_unique_problem))
    print(is_unique(not_unique_problem))
