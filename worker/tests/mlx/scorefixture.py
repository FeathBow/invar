import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from worker.mlx.scoring import arguments, run
from worker.tests.mlx.scoring import load

if __name__ == "__main__":
    run(arguments(), loader=load)
