from typing import Optional, Tuple

from tqdm import trange

from calc_puzzle.solver import solve, is_unique
from calc_puzzle.structs import Block, Problem
from calc_puzzle.visualize import visualize_problem, visualize_answer

import random
import numpy as np
import pulp
import copy


def create_problem(size: int, num_blocks: int, seed_size: int = 5) -> Optional[Tuple[Problem, np.ndarray]]:
    for _ in trange(1000):
        answer_board = create_random_answer_board(size, seed_size)
        for _ in trange(30):
            blocks, _ = split_answer_board(num_blocks, answer_board)
            problem = Problem(size, blocks)
            try:
                if is_unique(problem):
                    return problem, answer_board
            except AssertionError:
                continue

    return None


def create_random_answer_board(size, seed_size=5):
    all_numbers = set(range(0, size))
    vertical = [set([]) for _ in range(size)]
    horizontal = [set([]) for _ in range(size)]

    answer_board = None
    s = -1
    while s != 1:
        positions = [(x, y) for y in range(size) for x in range(size)]
        random.shuffle(positions)

        answer_board = -np.ones((size, size), dtype=np.int32)

        for x, y in positions[:seed_size]:
            l = list(all_numbers - (vertical[x] | horizontal[y]))
            if len(l) == 0:
                continue
            v = random.choice(list(all_numbers - (vertical[x] | horizontal[y])))

            answer_board[y, x] = v
            vertical[x].add(v)
            horizontal[y].add(v)

        answer_board[answer_board != -1] += 1
        prob = pulp.LpProblem("AnswerBoard", pulp.LpMinimize)
        prob += 0

        size = size
        numbers = range(1, size + 1)  # [1, 6)
        xs = range(1, size + 1)
        ys = range(1, size + 1)

        choices = pulp.LpVariable.dicts("Cell", (numbers, xs, ys), 0, 1, pulp.LpInteger)

        # 1つのマスに入る値は1つだけ

        for y in ys:
            for x in xs:
                prob += pulp.lpSum([choices[v][x][y] for v in numbers]) == 1

        for v in numbers:
            for y in ys:
                prob += pulp.lpSum([choices[v][x][y] for x in xs]) == 1

        for v in numbers:
            for x in xs:
                prob += pulp.lpSum([choices[v][x][y] for y in ys]) == 1

        for y in range(size):
            for x in range(size):
                if answer_board[y, x] != -1:
                    prob += choices[answer_board[y, x]][x + 1][y + 1] == 1

        s = prob.solve()
        answer_board = np.zeros((size, size), dtype=np.int32)
        for y in ys:
            for x in xs:
                for v in numbers:
                    if choices[v][x][y].value() == 1:
                        answer_board[y - 1][x - 1] = v
    return answer_board


def split_answer_board(num_centroids, answer_board):
    size = answer_board.shape[0]
    positions = [(x, y) for y in range(size) for x in range(size)]
    shuffled_positions = copy.deepcopy(positions)
    random.shuffle(shuffled_positions)
    centroids = [(random.uniform(-1, size), random.uniform(-1, size)) for _ in range(num_centroids)]

    blocks = [[0, []] for _ in range(num_centroids)]

    for x, y in positions:
        r = np.array([md((x, y), c) for c in centroids]).argmin()
        b = blocks[r]
        b[0] += answer_board[y, x]
        b[1].append((x + 1, y + 1))
    blocks = list(map(lambda x: Block(*x), blocks))

    return blocks, centroids


def md(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    problem,ab = create_problem(5, 8)
    print(problem)
    fig = visualize_problem(problem)
    fig.savefig("test_problem.png")
    fig = visualize_answer(problem, ab)
    fig.savefig("test_answer.png")
