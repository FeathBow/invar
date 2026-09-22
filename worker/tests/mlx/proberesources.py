import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from worker.mlx import adapter, model, score, tensors
from worker.tests.mlx.crossscore import loaded
from worker.tests.mlx.scoring import load


def prepare(directory, *, seed, source):
    directory.mkdir()
    (directory / "fixture.json").write_text(json.dumps({"seed": seed}))
    runtime = loaded(seed)
    identities = model.identities(runtime)
    checkpoint = directory / "adapter.safetensors"
    tensors.save_policy(checkpoint, adapter.state(runtime.model))
    previous = json.loads(source.read_text())
    inspection = directory / "source.json"
    inspection.write_text(previous["source_inspection"])
    configuration = directory / "config.json"
    configuration.write_text(json.dumps({"format": "invar-mlx-runtime-v1", "batch_size": 1,
                                         "prefill_step": 2, "cache_bytes": 1048576}))
    options = dict(path=str(inspection), cache=str(directory), adapter=str(checkpoint), config=str(configuration),
                   numerics="primary", probe_steps=json.dumps(previous["full_vocabulary"]["steps"]),
                   digest=identities["adapter"], **{key + "_digest": identities[key] for key in ("tokenizer", "base", "assembly")})
    manifest = directory / "options.json"
    manifest.write_text(json.dumps(options))
    print(json.dumps({"options": str(manifest), "identities": identities, "seed": seed}), flush=True)


def fixture_load(cache, *, numerics, **arguments):
    runtime = load(cache, **arguments)
    if runtime.numerics != numerics:
        raise ValueError("Producer fixture numerical profile differs")
    return runtime


def produce(path):
    value = json.loads(path.read_text())
    options = {key: Path(item) if key in ("path", "cache", "adapter", "config") else item for key, item in value.items()}
    score.run(SimpleNamespace(**options), loader=fixture_load)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--directory", type=Path, required=True)
    preparation.add_argument("--seed", type=int, required=True)
    preparation.add_argument("--source", type=Path, required=True)
    producer = commands.add_parser("produce")
    producer.add_argument("--options", type=Path, required=True)
    options = parser.parse_args()
    if options.action == "prepare":
        prepare(options.directory, seed=options.seed, source=options.source)
    else:
        produce(options.options)


if __name__ == "__main__":
    main()
