import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import time

from worker import core
from worker.baseline.measure import Services, flags, write
from worker.baseline import data as product_data
from worker.baseline import policy as product_policy


@dataclass(frozen=True, kw_only=True)
class Options:
    backend: str
    cache: Path
    initial: Path
    trained: Path | None = None
    trained_policy: str | None = None
    tasks: Path
    settings: Path
    configuration: Path
    output: Path
    mode: str = "paired"
    core: str = "invar"
    inference_python: str | None = None
    inference_entry: Path | None = None


class CudaInference:
    def __init__(self, client, *, settings, identity):
        self.client = client
        self.settings = dict(settings)
        self.identity = identity

    def generate(self, tasks):
        return self.client.exchange("generate", {"tasks": tasks})["results"]

    def activate(self, directory, *, settings):
        actual = self.client.exchange("activate", {"adapter": str(directory / "adapter.safetensors"),
                                                   "policy": settings["policy"]})["identities"]
        if actual != product_data.identities(settings, behavior=True):
            raise ValueError("Native evaluation activated a different policy or model profile")
        self.settings = dict(settings)


@contextmanager
def native_backend(options, *, settings, emit):
    if options.backend == "mlx":
        from worker.baseline.mlx import load
        from worker.mlx.model import load_native as model
        with load(options, settings=settings, emit=emit, loader=model) as backend:
            yield backend
        return
    if options.backend != "cuda":
        raise ValueError("Expected the explicit mlx or cuda native evaluation backend")
    from worker.baseline.process import open_owner
    with open_owner(options, settings=settings) as (client, ready):
        if ready["identities"] != product_data.identities(settings, behavior=True):
            raise ValueError("Native evaluation differs from the declared product model profile")
        yield CudaInference(client, settings=settings, identity=(ready["model"], ready["revision"]))


def generate(backend, workload, path):
    with path.open("x") as stream:
        for cohort in workload:
            for value in backend.generate(cohort["tasks"]):
                stream.write(json.dumps(value, allow_nan=False) + "\n")
            stream.flush()


def validate_mode(options):
    if options.mode not in ("paired", "initial"):
        raise ValueError("Expected paired or initial native evaluation mode")
    supplied = (options.trained, options.trained_policy)
    if options.mode == "paired" and any(value is None for value in supplied):
        raise ValueError("Paired native evaluation requires a trained checkpoint and policy")
    if options.mode == "initial" and any(value is not None for value in supplied):
        raise ValueError("Initial-only native evaluation does not select a trained policy")


def admit(options, services, settings):
    fields = {**settings, "tasks": options.tasks}
    if options.mode == "initial":
        kind, name = "native-evaluate", "evaluation"
        fields["log"] = options.output / "initial.jsonl"
    else:
        kind, name = "native-quality", "quality"
        fields.update({"trained-policy": options.trained_policy, "initial-log": options.output / "initial.jsonl",
                       "trained-log": options.output / "trained.jsonl"})
    observed = services.invoke(["replay", kind, *flags(fields)], executable=options.core)
    write(options.output / (name + ".json"), observed)
    return {name: observed}


def run(options, services):
    validate_mode(options)
    settings = core.decode(options.settings.read_text())
    planned = services.invoke(["replay", "native-plan", *flags({**settings, "tasks": options.tasks})], executable=options.core)
    if planned["format"] != "invar-native-composition-v1":
        raise ValueError("Unexpected native evaluation plan")
    options.output.mkdir(exist_ok=False)
    write(options.output / "plan.json", planned)
    with (options.output / "events.jsonl").open("x") as events:
        def emit(stage, values):
            events.write(json.dumps({"stage": stage, **values}, allow_nan=False) + "\n")
            events.flush()

        started = services.clock()
        initial = product_policy.read(options.initial, services=services, executable=options.core)
        product_policy.check_selection(initial, settings)
        with services.backend(options, settings=settings, emit=emit) as backend:
            if backend.settings != settings:
                raise ValueError("Native evaluation changed the declared product settings")
            product_policy.check_loaded(initial, backend)
            generate(backend, planned["workload"], options.output / "initial.jsonl")
            if options.mode == "paired":
                selected = product_policy.read(options.trained, services=services, executable=options.core)
                current = {**settings, "policy": options.trained_policy}
                product_policy.check_selection(selected, current)
                backend.activate(options.trained, settings={**current, "policy": selected["adapter"]})
                product_policy.check_loaded(selected, backend)
                generate(backend, planned["workload"], options.output / "trained.jsonl")
        seconds = services.clock() - started
    observed = admit(options, services, settings)
    result = {"backend": options.backend, "mode": options.mode, "execution_seconds": seconds, **observed,
              "scope": "independent native generation with no learner update; execution includes backend load through close; evaluation costs excluded from training-cycle timing; not numerical qualification or statistical generalization"}
    write(options.output / "complete.json", result)
    return result


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mlx", "cuda"), required=True)
    parser.add_argument("--mode", choices=("paired", "initial"), default="paired")
    parser.add_argument("--core", default="invar")
    parser.add_argument("--inference-python")
    parser.add_argument("--inference-entry", type=Path)
    parser.add_argument("--trained-policy")
    parser.add_argument("--trained", type=Path)
    for name in ("cache", "initial", "tasks", "settings", "configuration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    return Options(**vars(parser.parse_args()))


def main():
    print(json.dumps(run(arguments(), Services(backend=native_backend, invoke=core.invoke, clock=time.perf_counter)), allow_nan=False))


if __name__ == "__main__":
    main()
