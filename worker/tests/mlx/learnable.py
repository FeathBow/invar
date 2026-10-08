import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from worker.tests.mlx.fixture import load, main

if __name__ == "__main__":
    main(partial(load, learnable=True))
