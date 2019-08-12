import argparse
from pathlib import Path

from calc_puzzle.builder import create_problem
from calc_puzzle.visualize import visualize_problem, visualize_answer
from tqdm import trange


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("num_problems", type=int)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--size", "-s", type=int, default=5)
    parser.add_argument("--num-blocks", "-b", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)

    for i in trange(args.num_problems):
        problem, answer = create_problem(args.size, args.num_blocks)

        fig = visualize_problem(problem)
        fig.savefig(str(args.output / f"01_problem{i+1:03}.png"))
        fig = visualize_answer(problem, answer)
        fig.savefig(str(args.output / f"02_answer{i + 1:03}.png"))
