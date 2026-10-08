import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import json
import os
import subprocess
import sys

import torch

from worker.tests.hf.overlap import CHILD_SECONDS, CORE, execute, flags, inspected, journaled, prepared, workload

LOCK = "import fcntl, sys; held = open(sys.argv[1], 'a'); fcntl.lockf(held, fcntl.LOCK_EX); print(flush=True); sys.stdin.read()"


def bindings(recorded):
    return [tuple(entry["binding"].values()) for entry in recorded if entry["entry"] in ("dispatched", "attempt")]


def published(records):
    return {value["update"]: (value["policy"], value["learner"]) for value in records if value.get("phase") == "published"}


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.settings = prepared("invar-recovery-")

    def train(self, name, cycles, settings):
        records = execute([CORE, "train", *flags({**settings, "staleness": 1, "output": name})], self.root / (name + ".jsonl"),
                          stdin=workload(cycles), cwd=self.root)
        return published(records), self.root / name

    def resume(self, output, name):
        return execute([CORE, "train", "--resume", output], self.root / (name + ".jsonl"))

    def refused(self, output, problem):
        completed = subprocess.run([CORE, "train", "--resume", output], capture_output=True, text=True, timeout=CHILD_SECONDS)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(problem, completed.stderr)

    def cut(self, output, kept):
        journal = output / "journal.jsonl"
        lines = journal.read_text().splitlines(keepends=True)
        reserved = {entry["process"] for entry in map(json.loads, lines[:kept]) if entry["entry"] == "process"}
        for path in (output / "transcripts").iterdir():
            if int(path.stem) not in reserved:
                path.unlink()
        journal.write_text("".join(lines[:kept]) + '{"entry":"ev')

    def applied(self, output, update):
        recorded = journaled(output)
        call = [entry["binding"]["call"] for entry in recorded if entry["entry"] == "attempt" and entry["update"] == update][-1]
        receipt = [index for index, entry in enumerate(recorded) if entry["entry"] == "interval" and entry["role"] == "learner" and entry["update"] == update][-1]
        self.cut(output, receipt)
        (output / ("generation" + str(update + 1))).unlink()
        (output / ("staging" + str(call)) / "policy.json").unlink()
        return call, receipt

    def recorded(self, output, update):
        recorded = journaled(output)
        kept = [index for index, entry in enumerate(recorded) if entry["entry"] == "event" and entry["event"]["kind"] == "recorded" and entry["event"]["update"] == update][-1] + 1
        self.cut(output, kept)
        return kept

    def test_a_run_interrupted_before_its_first_publication_resumes_and_a_lost_acknowledgement_is_not_computed_again(self):
        relative = {**self.settings, "checkpoint": "initial", "reference": "initial/adapter.safetensors", "cache": "."}
        expected, output = self.train("early", 1, relative)
        with subprocess.Popen([sys.executable, "-c", LOCK, output / "journal.jsonl"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as holder:
            holder.stdout.readline()
            self.refused(output, "Another process holds the run journal")
            with self.assertRaisesRegex(ValueError, "Another process holds the run journal"):
                inspected(self.root, output, 1, "held")
            holder.stdin.close()
        history = inspected(self.root, output, 1, "early", process=True)
        self.assertEqual([attempt["outcome"] for attempt in history["training"]["attempts"]], [{"committed": 1}])
        self.applied(output, 0)
        records = self.resume(output, "earlyresumed")
        self.assertEqual(records[0], {"phase": "resumed", "committed": []})
        self.assertEqual(published(records), expected)
        kept = self.recorded(output, 0)
        names = sorted(path.name for path in output.iterdir())
        self.assertEqual(self.resume(output, "earlyacknowledged"), [{"phase": "resumed", "committed": [0]}])
        self.assertEqual(len(journaled(output)), kept + 1)
        self.assertEqual(sorted(path.name for path in output.iterdir()), names)
        with (output / "generation1" / "learner.pt").open("ab") as learner:
            learner.write(b"\0")
        self.refused(output, "Generation 1 differs from the update its attempt staged")
        (output / "generation1").unlink()
        self.refused(output, "Update 0 is committed in the journal but its generation is missing")
        self.refused(output.rename(output.with_name("moved")), "The declared output directory is not the run's directory")

    def test_an_update_applied_without_its_receipt_is_computed_again_and_committed_updates_are_kept(self):
        expected, output = self.train("interrupted", 2, self.settings)
        first = os.readlink(output / "generation1")
        abandoned, before = self.applied(output, 1)
        self.assertTrue((output / ("staging" + str(abandoned)) / "learner.pt").exists())
        records = self.resume(output, "resumed")
        self.assertEqual(records[0], {"phase": "resumed", "committed": [0]})
        self.assertEqual(published(records), {1: expected[1]})
        recorded = journaled(output)
        after = recorded[before:]
        self.assertEqual(after[0]["entry"], "restart")
        earlier = [entry["process"] for entry in recorded[:before] if entry["entry"] == "process"]
        self.assertGreater(min(entry["process"] for entry in after if entry["entry"] == "process"), max(earlier))
        self.assertEqual({entry["update"] for entry in after if entry["entry"] == "attempt"}, {1})
        self.assertEqual(os.readlink(output / "generation1"), first)
        self.assertNotEqual(os.readlink(output / "generation2"), "staging" + str(abandoned))
        identities = bindings(recorded)
        self.assertEqual(len(identities), len(set(identities)))
        native = torch.load(output / "generation2" / "learner.pt", weights_only=True)
        self.assertTrue(all(slot["step"].item() == 2 for slot in native["optimizer"]["state"].values()))
        history = inspected(self.root, output, 2, "resumed")
        self.assertEqual(len(history["artifacts"]), 2)
        outcomes = [(attempt["update"], attempt["outcome"]) for attempt in history["training"]["attempts"]]
        self.assertEqual(outcomes[0], (0, {"committed": 1}))
        self.assertEqual(outcomes[-1], (1, {"committed": 2}))
        self.assertTrue(any(update == 1 and "committed" not in outcome for update, outcome in outcomes[1:-1]))
        self.assertEqual(len(history["training"]["restarts"]), 1)


if __name__ == "__main__":
    unittest.main()
