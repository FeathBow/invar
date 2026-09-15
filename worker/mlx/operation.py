import argparse
from pathlib import Path

from worker.mlx.model import resolve
from worker.mlx.tokenization import digest
from worker.hf.operation import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    options = parser.parse_args()
    print(digest(load(resolve(options.cache))))


if __name__ == "__main__":
    main()
