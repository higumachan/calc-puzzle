from collections import namedtuple, defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from calc_puzzle.structs import Problem


def visualize_problem(problem: Problem):
    size = problem.size
    board_group = np.zeros((size, size), dtype=np.int32)
    board_sum_number = np.zeros((size, size), dtype=np.int32)

    for i, block in enumerate(problem.blocks):
        for x, y in block.positions:
            board_group[y-1, x-1] = i
            board_sum_number[y-1, x-1] = block.agg_number
    fig = plt.Figure()
    ax = fig.add_subplot()
    sns.heatmap(board_group, annot=board_sum_number, square=True, cbar=False, ax=ax, xticklabels=False, yticklabels=False)
    ax.set_ylim(size, 0)

    return fig


def visualize_answer(problem: Problem, answer_board):
    size = problem.size
    board_group = np.zeros((size, size), dtype=np.int32)

    for i, block in enumerate(problem.blocks):
        for x, y in block.positions:
            board_group[y - 1, x - 1] = i
    fig = plt.Figure()
    ax = fig.add_subplot()
    sns.heatmap(board_group, annot=answer_board, square=True, cbar=False, ax=ax, xticklabels=False,
                yticklabels=False)
    ax.set_ylim(size, 0)

    return fig
