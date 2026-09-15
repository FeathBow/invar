import argparse
from dataclasses import dataclass
from pathlib import Path

from worker import core
from worker.hf.codec import Session
from worker.compare import canonical, inputs


@dataclass(frozen=True, kw_only=True)
class Input:
    checkpoint: Path
    log: Path
    call: int


def compare(left, right, policy, *, core_executable="invar"):
    session = Session()
    arguments = ["compare", "states", "--codec-mode", "stdio", "--policy", policy,
                 *inputs(left, right, "checkpoint")]
    return core.exchange(arguments, executable=core_executable, handler=session.handle)


def arguments():
    parser = argparse.ArgumentParser(description="Compare native checkpoints through the Invar core")
    parser.add_argument("--core", default="invar", help="Core executable")
    parser.add_argument("--policy", type=Path, required=True, help="Actual common input adapter checkpoint")
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{side}-log", type=Path, required=True)
        parser.add_argument(f"--{side}-call", type=int, required=True)
    values = vars(parser.parse_args())
    sides = tuple(Input(**{field: values[f"{side}_{field}"] for field in ("checkpoint", "log", "call")})
                  for side in ("left", "right"))
    return sides, values["policy"], values["core"]


def main():
    sides, policy, executable = arguments()
    result = compare(*sides, policy, core_executable=executable)
    print(canonical(result))
    raise SystemExit(0 if result["equal"] else 1)


if __name__ == "__main__":
    main()
