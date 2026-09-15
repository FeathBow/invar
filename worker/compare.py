import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from worker import core
from worker import invocation

OBSERVATION = "objective and reward gradients before AdamW"
HEADER_BYTES = 8


@dataclass(frozen=True, kw_only=True)
class Claim:
    invocation: invocation.Invocation
    request: str
    digest: str
    result: str
    log: Path
    log_digest: str
    core: str


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def transported(value, path, executable):
    bound = value["invocation"]
    return Claim(invocation=invocation.Invocation(**bound["binding"], program=bound["program"]),
                 request=canonical(value["request"]), digest=value["digest"],
                 result=canonical(value["result"]), log=path,
                 log_digest=value["log_digest"], core=executable)


def claim(path, call, *, core_executable="invar"):
    value = core.invoke(["inspect", "update", "--log", path, "--call", call], executable=core_executable)
    return transported(value, path, core_executable)


def inputs(left, right, kind):
    result = []
    for side, value in (("left", left), ("right", right)):
        result.extend((f"--{side}-log", value.log, f"--{side}-call", value.call))
        if kind:
            result.extend((f"--{side}-{kind}", getattr(value, kind)))
    return result


def paired(left, right, *, core_executable="invar"):
    value = core.invoke(["inspect", "updates", *inputs(left, right, "")], executable=core_executable)
    return tuple(transported(value[name], source.log, core_executable)
                 for name, source in (("left", left), ("right", right)))


def compare(left, right, policy, *, core_executable="invar"):
    return core.invoke(["compare", "gradients", "--policy", policy, *inputs(left, right, "gradients")],
                       executable=core_executable)


@dataclass(frozen=True, kw_only=True)
class Input:
    gradients: Path
    log: Path
    call: int


def arguments():
    parser = argparse.ArgumentParser(description="Compare bound gradient observations through the Invar core")
    parser.add_argument("--core", default="invar", help="Core executable")
    parser.add_argument("--policy", type=Path, required=True)
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-gradients", type=Path, required=True)
        parser.add_argument(f"--{side}-log", type=Path, required=True)
        parser.add_argument(f"--{side}-call", type=int, required=True)
    values = vars(parser.parse_args())
    sides = tuple(Input(**{field: values[f"{side}_{field}"] for field in ("gradients", "log", "call")})
                  for side in ("left", "right"))
    return sides, values["policy"], values["core"]


def main():
    sides, policy, executable = arguments()
    result = compare(*sides, policy, core_executable=executable)
    print(canonical(result))
    raise SystemExit(0 if result["equal"] else 1)


if __name__ == "__main__":
    main()
