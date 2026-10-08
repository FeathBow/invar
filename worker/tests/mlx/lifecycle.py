import unittest

try:
    import mlx  # noqa: F401
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mlx.core as mx

from worker import core
from worker.mlx import checkpoint as mlx_checkpoint
from worker.mlx import codec as mlx_codec
from worker.mlx import tokenization as mlx_tokenization
from worker.tests.mlx.publication import seal
from worker.mlx import tensors as mlx_tensors
from worker.tests.mlx.rollout import tokenizer

ENTRY = Path(__file__).with_name("fixture.py").resolve()
LEARNABLE = Path(__file__).with_name("learnable.py").resolve()
CORE = os.environ.get("INVAR_CORE", "invar")


def flags(values):
    return [str(item) for key, value in values.items() for item in ("--" + key, value)]


class LifecycleTests(unittest.TestCase):
    def command(self, command, output, *, stdin=None):
        completed = subprocess.run(list(map(str, command)), input=stdin, capture_output=True, text=True, timeout=45)
        output.write_text(completed.stdout)
        output.with_suffix(".stderr").write_text(completed.stderr)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return [json.loads(line) for line in completed.stdout.splitlines()]

    def prepared(self, prefix, *, entry=ENTRY):
        root = Path(tempfile.mkdtemp(prefix=prefix))
        initial = root / "initial"
        log = root / "initial.jsonl"
        records = self.command([sys.executable, "-B", entry, *flags({"cache": root, "output": initial,
                               "seed": 17, "tokenizer-digest": mlx_tokenization.digest(tokenizer())})], log)
        observed = records[-1]
        inference = next(record["inference"] for record in records if record["stage"] == "profile")
        settings = {"policy": observed["policy"], "learner": observed["learner"], "reference-digest": observed["policy"],
                    **{name + "-digest": observed[name] for name in ("tokenizer", "base", "assembly")},
                    **{"behavior-" + name + "-digest": inference[name] for name in ("base", "assembly")},
                    "clip": 0.2, "penalty": 0, "delta": 0.0001, "rate": 0.0001,
                    "beta1": 0.9, "beta2": 0.999, "optimizer-epsilon": 0.00000001, "decay": 0}
        tasks = [{"tasks": [{"name": str(index), "group": "group", "prompt": "one two", "tokens": 2,
                              "seed": 17 + index, "temperature": 0.8, "answer": "#### 1"} for index in range(2)],
                  "order": [1, 0], "delivery": [0, 1]} for _ in range(2)]
        workload = root / "tasks.json"
        workload.write_text(json.dumps(tasks))
        configuration = root / "configuration.json"
        configuration.write_text(json.dumps({"format": "invar-mlx-runtime-v1", "batch_size": 2,
                                             "prefill_step": 16, "cache_bytes": 1048576}))
        selected = seal(root, initial, observed={**observed, **inference}, configuration=configuration, executable=CORE, worker=entry)
        return root, log, observed, inference, settings, workload, configuration, selected

    def trained(self, prepared, name, *, entry=ENTRY, settings=None, reference=None, **extra):
        root, _, _, _, declared, workload, configuration, _ = prepared
        settings = settings or declared
        shared = {"inference-mode": "shared", "learning-mode": "shared", "checkpoint": root / "initial",
                  "reference": reference or root / "initial" / "adapter.safetensors", "publication": "reference", "output": root / name}
        records = self.command([CORE, "train", *flags({**settings, **shared, **extra, "python": sys.executable,
                               "inference-python": sys.executable, "inference": entry, "learning": entry,
                               "inference-config": configuration, "cache": root})],
                               root / (name + ".jsonl"), stdin=workload.read_text())
        return records, shared

    def final(self, root, checkpoint, name, *, entry=ENTRY):
        described = core.invoke(["policy", "inspect", "--checkpoint", checkpoint], executable=CORE)
        final = {"digest": described["adapter"], "tokenizer-digest": described["tokenizer"], "base-digest": described["base"],
                 "assembly-digest": described["assembly"], "prompt": "one two", "tokens": 2, "temperature": 0.8, "seed": 17,
                 **{name: 100 for name in ("call", "attempt", "instance")}}
        log = root / (name + "final.jsonl")
        inputs = {key: final[key] for key in ("prompt", "tokens", "temperature", "seed", "call", "attempt", "instance")}
        self.command([CORE, "infer", *flags({**inputs, "python": sys.executable, "worker": entry, "cache": root,
                      "checkpoint": checkpoint})], log)
        return {"final-log": log, "final-exit-code": 0, **{"final-" + key: value for key, value in final.items()}}

    def compared(self, left, right):
        fields = {**{"left-" + key: value for key, value in left.items()}, **{"right-" + key: value for key, value in right.items()}}
        return core.exchange(["compare", "histories", "--codec-mode", "stdio", *flags(fields)],
                             executable=CORE, handler=mlx_codec.Session().handle)

    def test_shared_runtime_at_staleness_zero_publishes_the_lockstep_history_and_resumes(self):
        prepared = self.prepared("invar-mlx-runtime-", entry=LEARNABLE)
        root, log, *_ = prepared
        initial = mx.load(str(root / "initial" / "adapter.safetensors"))
        mx.random.seed(5)
        shifted = {name: mx.random.normal(value.shape) * 0.05 if name.endswith("lora_b") else value for name, value in initial.items()}
        reference = root / "reference.safetensors"
        settings = {**prepared[4], "penalty": 0.1, "reference-digest": mlx_tensors.save_policy(reference, shifted)}
        evidence = {"rng-profile": "mlx", "initial-source": "initializer", "initial-log": log, "initial-exit-code": 0,
                    "initial-seed": 17, "profile-mode": "roles"}
        _, lockstep = self.trained(prepared, "lockstep", entry=LEARNABLE, settings=settings, reference=reference)
        records, runtime = self.trained(prepared, "runtime", entry=LEARNABLE, settings=settings, reference=reference, staleness=0)
        output = runtime["output"]
        published = [value for value in records if value.get("phase") == "published"]
        self.assertEqual([value["update"] for value in published], [0, 1])
        self.assertNotEqual(published[0]["policy"], settings["policy"])
        self.assertNotEqual(published[1]["policy"], published[0]["policy"])
        left = {**settings, **lockstep, "tasks": prepared[5], "log": root / "lockstep.jsonl", "sessions": 1, "exit-code": 0,
                **evidence, **self.final(root, lockstep["output"] / "generation2", "lockstep", entry=LEARNABLE)}
        comparison = self.compared(left, {"run": output, **evidence, **self.final(root, output / "generation2", "runtime", entry=LEARNABLE)})
        self.assertTrue(comparison["equal"])
        self.assertTrue(comparison["schedule_equal"])
        self.assertTrue(comparison["execution"]["declared_equal"])
        self.assertEqual([comparison["execution"][side]["recorded"]["source"] for side in ("left", "right")], ["training log", "run directory"])
        journal = output / "journal.jsonl"
        entries = [json.loads(line) for line in journal.read_text().splitlines()]
        self.assertEqual([entry["role"] for entry in entries if entry["entry"] == "process"], ["shared"])
        call = [entry["binding"]["call"] for entry in entries if entry["entry"] == "attempt" and entry["update"] == 1][-1]
        receipt = [index for index, entry in enumerate(entries) if entry["entry"] == "interval" and entry["role"] == "learner" and entry["update"] == 1][-1]
        journal.write_text("".join(line + "\n" for line in journal.read_text().splitlines()[:receipt]) + '{"entry":"ev')
        (output / "generation2").unlink()
        (output / ("staging" + str(call)) / "policy.json").unlink()
        resumed = self.command([CORE, "train", "--resume", output], root / "resumed.jsonl")
        self.assertEqual(resumed[0], {"phase": "resumed", "committed": [0]})
        self.assertEqual([entry["role"] for entry in map(json.loads, journal.read_text().splitlines()) if entry["entry"] == "process"], ["shared", "shared"])
        history = core.exchange(["inspect", "history", "--codec-mode", "stdio", *flags({"run": output, **evidence, **self.final(root, output / "generation2", "resumed", entry=LEARNABLE)})],
                                executable=CORE, handler=mlx_codec.Session().handle)
        self.assertEqual(len(history["artifacts"]), 2)
        self.assertEqual(len(history["training"]["restarts"]), 1)
        outcomes = [(attempt["update"], attempt["outcome"]) for attempt in history["training"]["attempts"]]
        self.assertEqual(outcomes[-1], (1, {"committed": 2}))
        self.assertTrue(any(update == 1 and "committed" not in outcome for update, outcome in outcomes[:-1]))
        again = self.compared(left, {"run": output, **evidence, **self.final(root, output / "generation2", "again", entry=LEARNABLE)})
        self.assertTrue(again["equal"])

    def test_one_physical_owner_two_updates_and_native_state_boundaries(self):
        prepared = self.prepared("invar-mlx-lifecycle-")
        root, log, observed, inference, settings, workload, configuration, selected = prepared
        initial = root / "initial"
        records, shared = self.trained(prepared, "train")
        output = shared["output"]
        stages = [value.get("stage") for value in records]
        self.assertEqual(stages.count("load"), 1)
        self.assertEqual(stages.count("activation"), 3)
        self.assertEqual(stages.count("released"), 4)
        self.assertEqual(stages.count("closed"), 1)
        self.assertEqual(records[-1]["owner"], {"role": "shared", "session": 0})
        self.assertEqual(records[-1]["groups"], 4)
        publications = [value for value in records if value.get("phase") == "published"]
        self.assertEqual(len(publications), 2)
        for publication in publications:
            description = core.invoke(["policy", "inspect", "--checkpoint", publication["checkpoint"]], executable=CORE)
            self.assertEqual(description, {**selected, "adapter": publication["policy"]})
        native = mlx_checkpoint.load((output / "generation2/learner.pt").read_bytes())
        self.assertEqual(native["optimizer"]["state"]["step"].item(), 2)
        final = {"digest": publications[-1]["policy"], "tokenizer-digest": observed["tokenizer"],
                 "base-digest": inference["base"], "assembly-digest": inference["assembly"],
                 "prompt": "one two", "tokens": 2, "temperature": 0.8, "seed": 17, "call": 6, "attempt": 6, "instance": 6}
        final_inputs = {name: final[name] for name in ("prompt", "tokens", "temperature", "seed", "call", "attempt", "instance")}
        self.command([CORE, "infer", *flags({**final_inputs, "python": sys.executable, "worker": ENTRY, "cache": root,
                      "checkpoint": output / "generation2"})], root / "final.jsonl")
        history = {**settings, **shared, "tasks": workload, "log": root / "train.jsonl", "sessions": 1, "exit-code": 0,
                   "rng-profile": "mlx", "initial-source": "initializer", "initial-log": log, "initial-exit-code": 0,
                   "initial-seed": 17, "profile-mode": "roles", "final-log": root / "final.jsonl", "final-exit-code": 0,
                   **{"final-" + key: value for key, value in final.items()}}
        complete = core.exchange(["inspect", "history", "--codec-mode", "stdio", *flags(history)],
                                 executable=CORE, handler=mlx_codec.Session().handle)
        self.assertEqual(len(complete["artifacts"]), 2)
        self.assertTrue(all(item["gradients"]["nonzero_reward_tensors"] == [] for item in complete["artifacts"]))
        self.assertEqual([item["policy"] for item in publications], [observed["policy"]] * 2)
        trace = {**settings, "checkpoint": initial, "inference-mode": "shared", "learning-mode": "shared", "tasks": workload,
                 "log": root / "train.jsonl", "sessions": 1, "exit-code": 0, "output": output, "publication": "reference"}
        admitted = core.invoke(["inspect", "trace", *flags(trace)], executable=CORE)
        initialization = {**settings, "checkpoint": initial, "rng-profile": "mlx", "initial-source": "initializer",
                          "initial-log": log, "initial-exit-code": 0, "initial-seed": 17}
        initialized = core.exchange(["inspect", "initial", "--codec-mode", "stdio", *flags(initialization)],
                                     executable=CORE, handler=mlx_codec.Session().handle)
        self.assertEqual(initialized["state"]["mlx_rng_bytes"], [8])
        pids = (root / "loads.txt").read_text().splitlines()
        self.assertEqual(len(pids), 4)
        self.assertEqual(len(set(pids)), 4)
        self.assertEqual(len(admitted["observation"]["closed"]), 1)
        changed = [*records[:-1], {**records[-1], "groups": 3}]
        altered = root / "wrong-close.jsonl"
        altered.write_text("\n".join(map(json.dumps, changed)) + "\n")
        with self.assertRaisesRegex(ValueError, "Resident acknowledgement differs"):
            core.invoke(["inspect", "trace", *flags({**trace, "log": altered})], executable=CORE)
        self.evaluate(root, workload, configuration)

    def evaluate(self, root, workload, configuration):
        evaluation = root / "evaluation.jsonl"
        options = {"python": sys.executable, "worker": ENTRY, "worker-mode": "resident", "cache": root,
                   "worker-config": configuration, "checkpoint": root / "initial"}
        evaluated = self.command([CORE, "evaluate", *flags(options)], evaluation, stdin=workload.read_text())
        self.assertEqual(evaluated[-1]["phase"], "evaluation_complete")
        self.assertEqual(sum(row.get("stage") == "load" for row in evaluated), 1)
        self.assertEqual(len(set((root / "loads.txt").read_text().splitlines())), 5)


if __name__ == "__main__":
    unittest.main()
