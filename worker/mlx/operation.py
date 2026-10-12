import argparse
from pathlib import Path

from worker.mlx.model import configuration, resolve
from worker.mlx.tokenization import digest
from worker.hf.operation import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    options = parser.parse_args()
    print(digest(load(resolve(options.cache, configuration(options.config).identity()))))


if __name__ == "__main__":
    main()
