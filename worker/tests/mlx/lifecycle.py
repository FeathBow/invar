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

from worker import core
from worker.mlx import checkpoint as mlx_checkpoint
from worker.mlx import codec as mlx_codec
from worker.mlx import tokenization as mlx_tokenization
from worker.tests.mlx.publication import seal
from worker.tests.mlx.rollout import tokenizer

ENTRY = Path(__file__).with_name("fixture.py").resolve()
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

    def test_one_physical_owner_two_updates_and_native_state_boundaries(self):
        root = Path(tempfile.mkdtemp(prefix="invar-mlx-lifecycle-"))
        initial = root / "initial"
        log = root / "initial.jsonl"
        observed = self.command([sys.executable, "-B", ENTRY, *flags({"cache": root, "output": initial,
                                "seed": 17, "tokenizer-digest": mlx_tokenization.digest(tokenizer())})], log)[-1]
        settings = {"policy": observed["policy"], "learner": observed["learner"], "reference-digest": observed["policy"],
                    **{name + "-digest": observed[name] for name in ("tokenizer", "base", "assembly")},
                    **{"behavior-" + name + "-digest": observed[name] for name in ("base", "assembly")},
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
        selected = seal(root, initial, observed=observed, configuration=configuration, executable=CORE)
        output = root / "train"
        shared = {"inference-mode": "shared", "learning-mode": "shared", "checkpoint": initial,
                  "reference": initial / "adapter.safetensors", "publication": "reference", "output": output}
        records = self.command([CORE, "train", *flags({**settings, **shared, "python": sys.executable,
                               "inference-python": sys.executable, "inference": ENTRY, "learning": ENTRY,
                               "inference-config": configuration, "cache": root})],
                               root / "train.jsonl", stdin=workload.read_text())
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
                 "base-digest": observed["base"], "assembly-digest": observed["assembly"],
                 "prompt": "one two", "tokens": 2, "temperature": 0.8, "seed": 17, "call": 6, "attempt": 6, "instance": 6}
        final_inputs = {name: final[name] for name in ("prompt", "tokens", "temperature", "seed", "call", "attempt", "instance")}
        self.command([CORE, "infer", *flags({**final_inputs, "python": sys.executable, "worker": ENTRY, "cache": root,
                      "checkpoint": output / "generation2"})], root / "final.jsonl")
        history = {**settings, **shared, "tasks": workload, "log": root / "train.jsonl", "sessions": 1, "exit-code": 0,
                   "rng-profile": "mlx", "initial-source": "initializer", "initial-log": log, "initial-exit-code": 0,
                   "initial-seed": 17, "profile-mode": "roles", "final-log": root / "final.jsonl", "final-exit-code": 0,
                   **{"final-" + key: value for key, value in final.items()}}
        # Equal rewards and unchanged parameters form a legal history. Their
        # acceptance supplies no nondegenerate reward-learning evidence.
        complete = core.exchange(["inspect", "history", "--codec-mode", "stdio", *flags(history)],
                                 executable=CORE, handler=mlx_codec.Session().handle)
        self.assertEqual(len(complete["artifacts"]), 2)
        self.assertTrue(all(item["gradients"]["nonzero_reward_tensors"] == [] for item in complete["artifacts"]))
        self.assertEqual([item["policy"] for item in publications], [observed["policy"]] * 2)
        trace = {**settings, "inference-mode": "shared", "learning-mode": "shared", "tasks": workload,
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
        replay, = self.command([sys.executable, "-B", ENTRY.parents[3] / "entries" / "redo.py", *flags({
                               "python": sys.executable, "core": CORE, "worker": ENTRY, "mode": "shared", "cache": root,
                               "worker-config": configuration,
                               "initial": initial, "reference": initial / "adapter.safetensors", "log": root / "train.jsonl",
                               "output": root / "direct", "updates": 2, "exit-code": 0})], root / "direct.jsonl")
        self.assertEqual((replay["updates"], replay["equal_results"], replay["loads"], replay["sessions"]), (2, 2, 1, 1))
        self.assertEqual(len(set((root / "loads.txt").read_text().splitlines())), 5)
        self.evaluate(root, observed, workload, configuration)
        self.cycle(root, initial, trace, configuration)
        print(f"Native shared lifecycle artifacts: {root}", file=sys.stderr)

    def cycle(self, root, initial, trace, configuration):
        declaration = root / "cycle-reference.json"
        declaration.write_text(json.dumps({name: str(value) for name, value in trace.items()}))
        replay, = self.command([sys.executable, "-B", ENTRY.parents[3] / "entries" / "cycle.py", *flags({
                               "core": CORE, "python": sys.executable, "inference-python": sys.executable,
                               "inference": ENTRY, "learning": ENTRY, "cache": root, "initial": initial,
                               "reference": initial / "adapter.safetensors", "trace": declaration,
                               "inference-config": configuration, "output": root / "direct-cycle"})],
                               root / "direct-cycle.jsonl")
        self.assertTrue(replay["equal"])
        self.assertEqual((replay["sessions"], replay["publications"], len(replay["cycles"])), (1, 2, 2))
        self.assertEqual(len(set((root / "loads.txt").read_text().splitlines())), 8)

    def evaluate(self, root, observed, workload, configuration):
        adapter = root / "initial/adapter.safetensors"
        evaluation = root / "evaluation.jsonl"
        options = {"python": sys.executable, "worker": ENTRY, "worker-mode": "resident", "cache": root,
                   "worker-config": configuration, "checkpoint": root / "initial"}
        evaluated = self.command([CORE, "evaluate", *flags(options)], evaluation, stdin=workload.read_text())
        self.assertEqual(evaluated[-1]["phase"], "evaluation_complete")
        self.assertEqual(sum(row.get("stage") == "load" for row in evaluated), 1)
        direct, = self.command([sys.executable, "-B", ENTRY.parents[3] / "entries" / "direct.py", *flags({
                              "python": sys.executable, "core": CORE, "worker": ENTRY, "mode": "resident", "cache": root,
                              "worker-config": configuration, "adapter": adapter, "policy": observed["policy"],
                              "tasks": workload, "reference-log": evaluation, "reference-exit-code": 0,
                              "output": root / "direct-inference"})], root / "direct-inference.jsonl")
        self.assertEqual((direct["calls"], direct["equal_results"], direct["loads"]), (4, 4, 1))
        self.assertEqual(len(set((root / "loads.txt").read_text().splitlines())), 7)


if __name__ == "__main__":
    unittest.main()
