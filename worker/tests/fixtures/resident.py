import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from worker import registry

FORMAT = "invar-resident-v1"
BATCH = "invar-inference-batch-v1"
PROFILE_BYTES = 128 * 1024
FAILURE_STATUS = 19
MODEL = {"model": "resident-protocol-fixture", "revision": "fixed-fixture"}
TIMES = {"load": 0.01, "activation": 0.005, "inference": 0.02, "released": 0.0025, "closed": 0.00125}


def encoded(value):
    return json.dumps(value, allow_nan=False) + "\n"


def emit(value):
    raw = encoded(value)
    sys.stdout.write(raw)
    sys.stdout.flush()
    return raw.encode()


def timer(stage):
    return {"stage": stage, "cpu_seconds": TIMES[stage]}


def ready(call):
    identities = {key: call[key] for key in registry.FIELDS}
    loaded = {"stage": "loaded_adapter", "binding": call["binding"], "requested": call["adapter"],
              "consumed": call["adapter"], "load": call["load"], "image": registry.image(identities),
              **{key: call[key] for key in ("tokenizer", "base", "assembly")}, **MODEL, "scope": "protocol fixture"}
    return encoded(loaded) + encoded({"stage": "consumed", **call})


def result(call, *, fault):
    value = {"stage": "result", "binding": call["binding"], "request": call["request"],
             **{key: call[key] for key in registry.FIELDS}, "tokens": [11, 13, 17, 23], "prompt_length": 2,
             "behavior": [-1.0, -0.0], "behavior_bits": [3212836864, 2147483648],
             "text": "fixture #### 12", "truncated": True}
    if fault == "numerical":
        value = {**value, "behavior": [-0.5, -0.0], "behavior_bits": [3204448256, 2147483648]}
    return encoded(value)


def group(value, *, owner, count, fault):
    assert value["format"] == BATCH
    calls = [json.loads(raw) for raw in value["calls"]]
    digest = hashlib.sha256()
    before = [{"stage": "loading"}, {"stage": "profile", **MODEL, "precision": "explicit protocol fixture",
                                    "large_pipe_payload": "x" * PROFILE_BYTES}, timer("load")] if count == 0 else [timer("activation")]
    if count == 0 and fault == "profile":
        before = [{**row, "precision": "different protocol fixture"} if row["stage"] == "profile" else row for row in before]
    for record in [*before, {"stage": "consumed", "format": BATCH, "calls": [ready(call) for call in calls]}]:
        digest.update(emit(record))
    permitted = json.loads(sys.stdin.readline())
    assert permitted["format"] == BATCH
    assert [json.loads(raw) for raw in permitted["permissions"]] == [{key: call[key] for key in ("binding", "program")} for call in calls]
    for record in [timer("inference"), {"stage": "result", "format": BATCH, "calls": [result(call, fault=fault) for call in calls]}]:
        digest.update(emit(record))
    released = {"format": FORMAT, "owner": owner, "loads": [call["load"] for call in calls], "result_sha256": digest.hexdigest()}
    assert json.loads(sys.stdin.readline()) == {"action": "release", **released}
    if fault == "release":
        released = {**released, "loads": []}
    emit({"stage": "released", **released, "measurement": encoded(timer("released"))})


def serve(options):
    owner = {"role": "inference", "session": options.session}
    fault = json.loads(options.config.read_text())["fault"]
    emit_error = {"pid": os.getpid(), "owner": options.session, "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "adapter_argument": any(arg.startswith("--adapter=") for arg in sys.argv)}
    print(json.dumps(emit_error), file=sys.stderr, flush=True)
    if fault == "exit" and options.session == 0:
        raise SystemExit(FAILURE_STATUS)
    count = 0
    while True:
        value = json.loads(sys.stdin.readline())
        if value.get("format") == FORMAT:
            assert value == {"format": FORMAT, "owner": owner, "action": "close"}
            emit({"stage": "closed", "format": FORMAT, "owner": owner, "groups": count,
                  "measurement": encoded(timer("closed"))})
            assert not sys.stdin.readline()
            if fault == "late_exit":
                raise SystemExit(FAILURE_STATUS)
            return
        group(value, owner=owner, count=count, fault=fault)
        count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", type=Path, required=True)
    serve(parser.parse_args())


if __name__ == "__main__":
    main()
