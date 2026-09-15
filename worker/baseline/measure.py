import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

from worker.cohort import decode as cohort
from worker import core
from worker.baseline import policy as product_policy

INDEX_WIDTH = 4
FORMAT = "invar-native-composition-v1"


@dataclass(frozen=True, kw_only=True)
class Options:
    backend: str
    cache: Path
    initial: Path
    reference: Path
    tasks: Path
    settings: Path
    configuration: Path
    output: Path
    publication: str
    core: str = "invar"
    inference_python: str | None = None
    inference_entry: Path | None = None
    gradient_observation: str = "objective"


@dataclass(frozen=True, kw_only=True)
class Services:
    backend: object
    invoke: object
    clock: object


def flags(values):
    return [str(item) for name, value in values.items() for item in ("--" + name, value)]


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")


def run(options, services):
    settings = core.decode(options.settings.read_text())
    planned = services.invoke(["replay", "native-plan", *flags({**settings, "tasks": options.tasks})], executable=options.core)
    if planned["format"] != FORMAT:
        raise ValueError("Unexpected native composition plan")
    options.output.mkdir(exist_ok=False)
    checkpoints = options.output / "checkpoints"
    checkpoints.mkdir()
    write(options.output / "plan.json", planned)
    with (options.output / "events.jsonl").open("x") as events:
        def emit(stage, values):
            events.write(json.dumps({"stage": stage, **values}, allow_nan=False) + "\n")
            events.flush()

        started = services.clock()
        selected = product_policy.read(options.initial, services=services, executable=options.core)
        product_policy.check_selection(selected, settings)
        with services.backend(options, settings=settings, emit=emit) as backend:
            product_policy.check_loaded(selected, backend)
            current = backend.settings
            write(options.output / "materialization.json", current)
            results = []
            for index, workload in enumerate(planned["workload"]):
                cycle_started = started if index == 0 else services.clock()
                result, current, selected = advance(options, services, (backend, current, selected),
                                          selection=(index, workload), emit=emit, started=cycle_started)
                results.append(result)
        seconds = services.clock() - started
    result = {"format": FORMAT, "backend": options.backend, "cycles": results,
              "gradient_observation": options.gradient_observation,
              "execution_seconds": seconds, "publications": len(results),
              "response_tokens": sum(value["response_tokens"] for value in results),
              "final": current,
              "scope": "native API composition with resident models and optimizer, actual core-scored output, durable publication and published adapter activation; no live Invar evidence or exact-profile comparison"}
    write(options.output / "complete.json", result)
    return result


def advance(options, services, state, *, selection, emit, started):
    backend, settings, selected = state
    index, workload = selection
    directory = options.output / f"{index:0{INDEX_WIDTH}d}"
    directory.mkdir()
    generated = backend.generate(workload["tasks"])
    observed = directory / "inference.jsonl"
    with observed.open("x") as stream:
        for value in generated:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
    actual = services.invoke(["replay", "native-input", *flags({**settings, "tasks": options.tasks,
                              "cohort": index, "log": observed})], executable=options.core)
    write(directory / "input.json", actual)
    request = cohort(actual["request"])
    staging = f"stage{index + 1}"
    destination = f"generation{index + 1}"
    output = options.output / "checkpoints" / staging
    output.mkdir()
    updated = backend.update(request, output)
    write(directory / "update.json", updated)
    if updated["before"] != settings["policy"]:
        raise RuntimeError("Native update consumed a different policy")
    policy = {**selected, "adapter": updated["policy"]}
    receipt = services.invoke(["replay", "publish", *flags({"output": output.parent, "staging": staging,
                              "destination": destination, "publication": options.publication})],
                             executable=options.core, stdin=json.dumps(policy, allow_nan=False))
    write(directory / "publication.json", receipt)
    published = product_policy.successor(output.parent / destination, policy, services=services, executable=options.core)
    current = {**settings, "policy": published["adapter"], "learner": updated["learner"]}
    backend.activate(output.parent / destination, settings=current)
    product_policy.check_loaded(published, backend)
    result = {"index": index, "seconds": services.clock() - started,
              "response_tokens": sum(len(value["behavior_bits"]) for value in generated),
              "rewards": actual["rewards"], "update": updated, "checkpoint": str(output.parent / destination)}
    emit("cycle", result)
    return result, current, published


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mlx", "cuda"), required=True)
    parser.add_argument("--core", default="invar")
    parser.add_argument("--inference-python", help="Existing vLLM environment for the CUDA inference owner")
    parser.add_argument("--inference-entry", type=Path, help="Explicit native environment entry that calls product_inference.main")
    parser.add_argument("--publication", choices=("reference", "rename"), required=True)
    parser.add_argument("--gradient-observation", choices=("objective", "objective-and-reward"), default="objective",
                        help="Record a second parameter VJP for reward-learning feasibility; its cost remains included")
    for name in ("cache", "initial", "reference", "tasks", "settings", "configuration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    return Options(**vars(parser.parse_args()))


def native_backend(options, *, settings, emit):
    if options.backend == "mlx":
        from worker.baseline.mlx import load
        from worker.mlx.model import load_native as model
        return load(options, settings=settings, emit=emit, loader=model)
    if options.backend != "cuda":
        raise ValueError("Expected the explicit mlx or cuda native backend")
    from worker.baseline.composed import load
    return load(options, settings=settings, emit=emit)


def main():
    print(json.dumps(run(arguments(), Services(backend=native_backend, invoke=core.invoke, clock=time.perf_counter)), allow_nan=False))


if __name__ == "__main__":
    main()
