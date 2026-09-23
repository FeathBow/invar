from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from worker import probe
from worker.distribution import FP32_BYTES
from worker.mlx import adapter, model, score, tensors
from worker.mlx.rollout import generate
from worker.tests.mlx.crossscore import SAMPLING, inspection, loaded, requests
from worker.tests.mlx.scoring import load
from worker.tests.probe import RELATIVE_CHECK_TOLERANCE, decimal_kl


class ProbeArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="invar-full-vocabulary-"))
        cls.reference = loaded(71)
        requested = requests()[1]
        generated, = generate(cls.reference.model, cls.reference.tokenizer, (requested,), sampling=SAMPLING)
        cls.source = cls.root / "source.json"
        cls.source.write_bytes(inspection(cls.reference, generated))
        cls.steps = (0, len(generated.behavior) - 1)
        if cls.steps[0] == cls.steps[1]:
            raise RuntimeError("The actual probe fixture needs distinct initial and later contexts")
        cls.config = cls.root / "config.json"
        cls.config.write_text(json.dumps({"format": "invar-mlx-runtime-v1", "batch_size": 1,
                                        "prefill_step": 2, "cache_bytes": 1048576}))
        cls.artifacts = []
        for seed in (71, 97):
            runtime = loaded(seed)
            identities = model.identities(runtime)
            cache = cls.root / str(seed)
            cache.mkdir()
            (cache / "fixture.json").write_text(json.dumps({"seed": seed}))
            checkpoint = cache / "adapter.safetensors"
            tensors.save_policy(checkpoint, adapter.state(runtime.model))
            options = SimpleNamespace(path=cls.source, cache=cache, adapter=checkpoint, config=cls.config,
                                      numerics="primary", probe_steps=json.dumps(cls.steps), digest=identities["adapter"],
                                      **{key + "_digest": identities[key] for key in ("tokenizer", "base", "assembly")})

            def loader(cache, *, numerics, **arguments):
                if numerics != runtime.numerics:
                    raise ValueError("Unexpected fixture numerical profile")
                return load(cache, **arguments)

            output = io.StringIO()
            with redirect_stdout(output):
                score.run(options, loader=loader)
            path = cls.root / (str(seed) + "-probe.json")
            path.write_text(output.getvalue())
            cls.artifacts.append(probe.read(path.read_bytes()))

    def test_actual_checkpoint_probes_compare_every_selected_vocabulary_coordinate(self):
        result = probe.compare(*self.artifacts)
        self.assertEqual(result["steps"], list(self.steps))
        self.assertEqual(result["vocabulary"], self.reference.config["vocab_size"])
        self.assertEqual(result["use_admission"], "not_evaluated")
        self.assertNotEqual(result["reference"]["target"]["base"], result["candidate"]["target"]["base"])
        for row, left, right in zip(result["observations"], self.artifacts[0].vectors.snapshots, self.artifacts[1].vectors.snapshots, strict=True):
            self.assertGreater(row["kl_reference_candidate"]["value"], 0)
            for direction, p, q in (("kl_reference_candidate", left, right), ("kl_candidate_reference", right, left)):
                expected = float(decimal_kl(p.probability_bits, q.probability_bits))
                self.assertTrue(math.isclose(row[direction]["value"], expected, rel_tol=RELATIVE_CHECK_TOLERANCE, abs_tol=0))
            self.assertEqual(row["prefix_length"], len(self.artifacts[0].source.path.prefix) + row["step"])
        for side in ("reference", "candidate"):
            self.assertGreater(result[side]["peak_active_bytes"], 0)
            self.assertGreater(result[side]["scoring_seconds"], 0)
            self.assertEqual(result[side]["raw_vector_bytes"], len(self.steps) * result["vocabulary"] * FP32_BYTES)
        own = probe.compare(self.artifacts[0], self.artifacts[0])
        self.assertTrue(all(row["kl_reference_candidate"]["value"] == 0 and row["total_variation"] == 0 for row in own["observations"]))

    def test_individually_valid_artifacts_still_require_identical_source_steps_and_vocabulary(self):
        original = json.loads(self.artifacts[1].encoded)
        source = deepcopy(original)
        source["source"]["log_sha256"] = "0" * 64
        source["source_inspection"] = json.dumps(source["source"])
        source["source_inspection_sha256"] = hashlib.sha256(source["source_inspection"].encode()).hexdigest()
        steps = deepcopy(original)
        steps["full_vocabulary"]["steps"] = [self.steps[-1]]
        steps["full_vocabulary"]["snapshots"] = [steps["full_vocabulary"]["snapshots"][-1]]
        steps["full_vocabulary"]["raw_payload_bytes"] //= len(self.steps)
        vocabulary = deepcopy(original)
        vocabulary["full_vocabulary"]["vocabulary"] += 1
        vocabulary["full_vocabulary"]["raw_payload_bytes"] += len(self.steps) * FP32_BYTES
        for snapshot in vocabulary["full_vocabulary"]["snapshots"]:
            snapshot["probability_bits"].append(0)
        for value in (source, steps, vocabulary):
            checked = probe.read(json.dumps(value).encode())
            with self.assertRaises(ValueError):
                probe.compare(self.artifacts[0], checked)

    def test_malformed_vectors_probability_roles_and_missing_costs_fail(self):
        original = json.loads(self.artifacts[0].encoded)
        changes = [
            lambda value: value["full_vocabulary"].update(snapshots=[]),
            lambda value: value["full_vocabulary"].update(steps=[0, 0]),
            lambda value: value["full_vocabulary"].update(vocabulary=True),
            lambda value: value["full_vocabulary"].update(raw_payload_bytes=0),
            lambda value: value["full_vocabulary"].update(representation="F32 log probabilities"),
            lambda value: value["full_vocabulary"]["snapshots"][0]["probability_bits"].pop(),
            lambda value: value["full_vocabulary"]["snapshots"][0]["probability_bits"].__setitem__(0, 0x7fc00000),
            lambda value: value["probability"].update(role="rl_reference"),
            lambda value: value["execution"].update(cache_origin="reference cache"),
            lambda value: value["target"].update(tokenizer="0" * 64),
            lambda value: value.update(measurements=[]),
            lambda value: value.update(source_inspection_sha256="0" * 64),
            lambda value: value["source"]["binding"].update(call=False),
            lambda value: value["prefix_tokens"].append(0),
        ]
        for index, change in enumerate(changes):
            value = deepcopy(original)
            change(value)
            self.assertNotEqual(json.dumps(value), json.dumps(original))
            with self.subTest(change=index), self.assertRaises(ValueError):
                probe.read(json.dumps(value).encode())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            probe.read(self.artifacts[0].encoded.rstrip()[:-1] + b', "format": "duplicate"}')


if __name__ == "__main__":
    unittest.main()
