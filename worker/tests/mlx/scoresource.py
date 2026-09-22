import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from worker.hf.infer import arguments
from worker.mlx.infer import run
from worker.tests.mlx.scoring import load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    run(arguments(parser=parser), loader=load)
