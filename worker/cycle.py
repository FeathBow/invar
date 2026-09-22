import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time

from worker import core
from worker import cycling as cycle_process
from worker import direct
from worker import partition as direct_resident


@dataclass(frozen=True, kw_only=True)
class Options:
    core: str
    python: str
    inference_python: str
    inference: Path
    learning: Path
    cache: Path
    initial: Path
    reference: Path
    trace: Path
    output: Path
    inference_config: Path | None = None
    devices: tuple[str, ...] | None = None

    def command(self, role):
        inference = role == "inference"
        python = self.inference_python if inference else self.python
        worker = self.inference if inference else self.learning
        command = [python, "-B", str(worker), f"--cache={self.cache}"]
        if not inference:
            command.append(f"--reference={self.reference}")
        if role != "learning" and self.inference_config is not None:
            command.append(f"--config={self.inference_config}")
        if role == "shared":
            command.append("--shared")
        return command


def flags(values):
    return [str(item) for name, value in values.items() for item in ("--" + name, value)]


def write(path, value):
    with path.open("x") as output:
        output.write(json.dumps(value, allow_nan=False) + "\n")


def declaration(path):
    value = core.decode(path.read_text())
    if not isinstance(value, dict) or any(not isinstance(name, str) or type(item) not in (str, int, float)
                                          for name, item in value.items()):
        raise ValueError("A cycle reference declaration maps core trace option names to scalar arguments")
    return value


def execution(options, services):
    trace = declaration(options.trace)
    reference = ["--initial", options.initial, *flags(trace)]
    planned = core.invoke(["replay", "plan", *reference], executable=options.core)
    cycle_process.policy(options, options.initial, expected=planned["cycles"][0]["policy"])
    if planned["mode"] == "shared" and (options.python != options.inference_python
                                        or options.inference.resolve() != options.learning.resolve()):
        raise ValueError("A shared cycle replay requires one selected Python and worker for both roles")
    placements = direct_resident.placements(options, planned["sessions"], environment=services.environment)
    if planned["mode"] == "shared" and options.devices is not None:
        raise ValueError("A native shared cycle uses its inherited physical device")
    options.output.mkdir()
    write(options.output / "reference.json", planned)
    write(options.output / "trace.json", trace)
    started = services.clock()
    measured = cycle_process.execute(options, services, planned=planned, tasks=trace["tasks"],
                                      placements=placements, flags=flags)
    return {"mode": planned["mode"], **measured}, reference, started


def execute(options, services):
    measured, _, started = execution(options, services)
    report = {**measured, "wall_seconds": services.clock() - started,
              "scope": "direct numerical cycle execution with actual semantic input preparation, original physical lifetimes, durable publication and own-successor consumption; offline output comparison has not run"}
    write(options.output / "execution.json", report)
    return report


def run(options, services):
    measured, reference, started = execution(options, services)
    checking = services.clock()
    observed = core.invoke(["replay", "inspect", *reference, *flags({"replay-output": options.output / "checkpoints",
                           "replay-log": options.output / "training.jsonl", "replay-exit-code": 0})], executable=options.core)
    write(options.output / "observations.json", observed)
    report = {**measured, "equal": observed["equal"],
              "comparison_seconds": services.clock() - checking, "wall_seconds": services.clock() - started,
              "scope": "direct complete numerical cycles with actual semantic input preparation, original physical lifetimes, durable publication and own-successor consumption; offline output comparison is reported separately; no live Invar execution or qualification authority"}
    write(options.output / "complete.json", report)
    return report


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", default="invar")
    for name in ("python", "inference-python"):
        parser.add_argument("--" + name, required=True)
    for name in ("inference", "learning", "cache", "initial", "reference", "trace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--inference-config", type=Path)
    parser.add_argument("--devices", type=lambda value: tuple(value.split(",")),
                        help="Resident inference devices in the original physical-owner order")
    return Options(**vars(parser.parse_args()))


def main(*, execution_only=False):
    services = direct.Services(run=subprocess.run, spawn=subprocess.Popen, clock=time.perf_counter,
                               environment=dict(os.environ))
    selected = execute if execution_only else run
    print(json.dumps(selected(arguments(), services), allow_nan=False))


if __name__ == "__main__":
    main()
