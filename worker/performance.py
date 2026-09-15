import argparse
import json
from pathlib import Path

from worker import evidence


def report(manifest, tasks, policy, *, core_executable="invar"):
    return evidence.performance(manifest, tasks, policy, core_executable=core_executable)


def main():
    parser = argparse.ArgumentParser(description="Summarize complete repeated Invar and direct inference measurements")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--core", default="invar", help="Core executable owning measurement admission and statistics")
    options = parser.parse_args()
    print(json.dumps(report(options.manifest, options.tasks, options.policy, core_executable=options.core), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
