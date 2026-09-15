import argparse
import json
from pathlib import Path

from worker import core as host
from worker import evaluation


def compare(definition, initial, trained, *, core="invar"):
    arguments = ["quality", "--tasks", definition]
    for side, run in (("initial", initial), ("trained", trained)):
        arguments.extend((f"--{side}-log", run.log, f"--{side}-policy", run.policy, f"--{side}-exit-code", run.exit_code))
    return host.invoke(arguments, executable=core)


def arguments():
    parser = argparse.ArgumentParser(description="Compare complete initial and trained-policy evaluation reports")
    parser.add_argument("--tasks", type=Path, required=True, help="Identical frozen input used by both evaluations")
    parser.add_argument("--core", default="invar", help="Invar executable that admits and compares the reports")
    for side in ("initial", "trained"):
        parser.add_argument(f"--{side}-log", type=Path, required=True, help="Complete invar evaluate stdout")
        parser.add_argument(f"--{side}-policy", required=True, help="Expected canonical adapter identity")
        parser.add_argument(f"--{side}-exit-code", type=int, required=True, help="Independently observed evaluation process exit status")
    values = vars(parser.parse_args())
    runs = tuple(evaluation.Run(**{field: values[f"{side}_{field}"] for field in ("log", "policy", "exit_code")})
                 for side in ("initial", "trained"))
    return {"definition": values["tasks"], "initial": runs[0], "trained": runs[1], "core": values["core"]}


def main():
    print(json.dumps(compare(**arguments()), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
