# How to use

## Install

```bash
git clone https://github.com/higumachan/calc-puzzle.git
cd calc-puzzle
pipenv --python 3.7 sync
```


## build problem

This sample is create 3 problems.

```bash
mkdir problems
env PYTHONPATH="." pipenv run python scripts/create_problems.py -o problems 3
```
