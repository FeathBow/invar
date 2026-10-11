import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import hashlib
import json

import torch
from safetensors.torch import save_file

from worker import float32
from worker.hf.score import scored
from worker.tests.hf.overlap import CORE, PROMPT, SEEDS, execute, flags, journaled, prepared, transcribed
from worker.tests.hf.scoring import MODELS, SCORE_ENTRY, TEACHER, identities

TEMPERATURE = 0.8


def workload(cycles):
    tasks = [{"name": str(index), "prompt": PROMPT, "tokens": 4, "seed": seed, "temperature": TEMPERATURE}
             for index, seed in enumerate(SEEDS)]
    return json.dumps([{"tasks": tasks, "order": [0, 1], "delivery": [1, 0]} for _ in range(cycles)])


class DistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, settings = prepared("invar-distillation-")
        cls.teacher = MODELS[TEACHER]()
        identity = identities(cls.teacher)
        adapter = cls.root / "teacher.safetensors"
        save_file({}, adapter)
        config = cls.root / "teacher.json"
        config.write_text(json.dumps({"model": TEACHER[0], "revision": TEACHER[1]}))
        cls.settings = {**{key: value for key, value in settings.items() if key != "delta"},
                        "algorithm": "opd", "teacher": SCORE_ENTRY, "teacher-adapter": adapter, "teacher-config": config,
                        "teacher-model": TEACHER[0], "teacher-revision": TEACHER[1], "teacher-digest": identity["adapter"],
                        "teacher-base-digest": identity["base"], "teacher-assembly-digest": identity["assembly"],
                        "teacher-profile-digest": hashlib.sha256(b"cpu fixture teacher profile").hexdigest(),
                        "teacher-temperature": TEMPERATURE}

    def test_each_rollout_completes_after_its_teacher_reading_and_the_update_learns_from_those_words(self):
        output = self.root / "distilled"
        records = execute([CORE, "train", *flags({**self.settings, "output": output, "staleness": 0})],
                          self.root / "distilled.jsonl", stdin=workload(2))
        self.assertEqual([(value["phase"], value["update"]) for value in records], [("published", 0), ("published", 1)])
        entries = journaled(output)
        teachers = {entry["process"] for entry in entries if entry["entry"] == "process" and entry["role"] == "teacher"}
        self.assertEqual({entry["process"]: entry["status"] for entry in entries if entry["entry"] == "exit" and entry["process"] in teachers},
                         {number: 0 for number in teachers})
        positions = {}
        for position, entry in enumerate(entries):
            if entry["entry"] in ("result", "teacher", "reading"):
                positions[entry["request"], entry["entry"]] = position
            elif entry["entry"] == "event" and entry["event"]["kind"] == "completed":
                positions[entry["event"]["request"], "completed"] = position
        readings = {entry["request"]: entry for entry in entries if entry["entry"] == "reading"}
        completions = {entry["event"]["request"]: entry["event"]["digest"] for entry in entries
                       if entry["entry"] == "event" and entry["event"]["kind"] == "completed"}
        self.assertEqual(sorted(readings), [0, 1, 2, 3])
        self.assertEqual(len(teachers), 4)
        for request, reading in readings.items():
            self.assertLess(positions[request, "result"], positions[request, "teacher"])
            self.assertLess(positions[request, "teacher"], positions[request, "reading"])
            self.assertLess(positions[request, "reading"], positions[request, "completed"])
            self.assertEqual(completions[request], reading["digest"])
        consumed = [value["request"] for value in transcribed(output, "learner") if value.get("stage") == "consumed" and "samples" in value["request"]]
        self.assertEqual([request["specification"] for request in consumed], ["opd-token-mean/v1"] * 2)
        samples = [sample for request in consumed for sample in request["samples"]]
        self.assertEqual(sorted(json.dumps(sample["teacher_call"], sort_keys=True) for sample in samples),
                         sorted(json.dumps(reading["binding"], sort_keys=True) for reading in readings.values()))
        for sample in samples:
            direct = scored(self.teacher, torch.tensor([sample["tokens"]]), sample["prompt_length"], TEMPERATURE, device="cpu")
            self.assertEqual(sample["teacher_bits"], [float32.word(value) for value in direct.tolist()])
            self.assertNotEqual(sample["teacher_bits"], sample["behavior_bits"])


if __name__ == "__main__":
    unittest.main()
