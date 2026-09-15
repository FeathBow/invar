import argparse
from dataclasses import dataclass
from pathlib import Path

from worker import core
from worker.compare import canonical, inputs


def snapshot(path, expected):
    return core.invoke(["inspect", "probabilities", "--probabilities", path,
                        "--log", expected.log, "--call", expected.invocation.call,
                        "--log-digest", expected.log_digest], executable=expected.core)["samples"]


@dataclass(frozen=True, kw_only=True)
class Input:
    probabilities: Path
    log: Path
    call: int


def compare(left, right, *, core_executable="invar"):
    return core.invoke(["compare", "probabilities", *inputs(left, right, "probabilities")],
                       executable=core_executable)


def arguments():
    parser = argparse.ArgumentParser(description="Compare bound probability words through the Invar core")
    parser.add_argument("--core", default="invar", help="Core executable")
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-probabilities", type=Path, required=True)
        parser.add_argument(f"--{side}-log", type=Path, required=True)
        parser.add_argument(f"--{side}-call", type=int, required=True)
    values = vars(parser.parse_args())
    sides = tuple(Input(**{field: values[f"{side}_{field}"] for field in ("probabilities", "log", "call")})
                  for side in ("left", "right"))
    return sides, values["core"]


def main():
    sides, executable = arguments()
    result = compare(*sides, core_executable=executable)
    print(canonical(result))
    raise SystemExit(0 if result["equal"] else 1)


if __name__ == "__main__":
    main()
