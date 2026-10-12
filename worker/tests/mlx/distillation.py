import unittest

try:
    import mlx  # noqa: F401
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import hashlib
import json
from pathlib import Path

from worker.implementation import INFERENCE
from worker.mlx import model, tensors
from worker.tests.mlx.crossscore import loaded
from worker.tests.mlx.lifecycle import CORE, Training
from worker.tests.mlx.scoring import SCORE_ENTRY, TEACHER_SEED

TEACHER = ("invar-nonuniform-hybrid-test", str(TEACHER_SEED))


class DistillationTests(Training, unittest.TestCase):
    def test_a_resident_teacher_scores_every_rollout_of_a_shared_run_and_the_run_resumes(self):
        prepared = self.prepared("invar-mlx-distillation-")
        root, _, _, _, settings, _, configuration, _ = prepared
        (root / "fixture.json").write_text(json.dumps({"seed": TEACHER_SEED, "adapted": False}))
        identity = model.identities(loaded(TEACHER_SEED, adapted=False), INFERENCE)
        adapter = root / "teacher.safetensors"
        tensors.save_policy(adapter, {})
        tasks = [{"tasks": [{"name": str(index), "prompt": "one two", "tokens": 2, "seed": 17 + index, "temperature": 0.8}
                            for index in range(2)], "order": [1, 0], "delivery": [0, 1]} for _ in range(2)]
        workload = root / "distillation.json"
        workload.write_text(json.dumps(tasks))
        distilled = {**{key: value for key, value in settings.items() if key != "delta"},
                     "algorithm": "opd", "teacher": SCORE_ENTRY, "teacher-adapter": adapter, "teacher-config": configuration,
                     "teacher-mode": "resident", "teacher-model": TEACHER[0], "teacher-revision": TEACHER[1],
                     "teacher-digest": identity["adapter"], "teacher-base-digest": identity["base"],
                     "teacher-assembly-digest": identity["assembly"],
                     "teacher-profile-digest": hashlib.sha256(b"mlx fixture teacher profile").hexdigest(),
                     "teacher-temperature": 0.8}
        records, shared = self.trained((*prepared[:5], workload, *prepared[6:]), "distilled", settings=distilled, staleness=0)
        self.assertEqual([value["update"] for value in records if value.get("phase") == "published"], [0, 1])
        output = shared["output"]
        entries = [json.loads(line) for line in (output / "journal.jsonl").read_text().splitlines()]
        teachers = [entry["process"] for entry in entries if entry["entry"] == "process" and entry["role"] == "teacher"]
        self.assertEqual(len(teachers), 1)
        self.assertEqual(sorted(entry["request"] for entry in entries if entry["entry"] == "reading"), [0, 1, 2, 3])
        self.assertEqual({entry["process"] for entry in entries if entry["entry"] == "teacher"}, set(teachers))
        stages = [json.loads(line)["stage"] for line in (output / "transcripts" / f"{teachers[0]}.jsonl").read_text().splitlines()]
        self.assertEqual((stages.count("score_result"), stages.count("released"), stages[-1]), (4, 4, "closed"))
        resumed = self.command([CORE, "train", "--resume", output], Path(root) / "resumed.jsonl")
        self.assertEqual(resumed, [{"phase": "resumed", "committed": [0, 1]}])


if __name__ == "__main__":
    unittest.main()
