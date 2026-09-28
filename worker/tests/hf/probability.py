import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import json
import tempfile
from pathlib import Path

import torch

from worker.hf.learning import update
from worker.hf.probability import FORMAT, save, words
from worker.tests.hf.learning import LOGICAL_ORDER, batch, make_learner

NEGATIVE_ZERO = 0x80000000


class ProbabilityTests(unittest.TestCase):
    def test_file_repeats_the_reported_steps_in_logical_order(self):
        logical = batch(steps=(("c",), ("a", "b"), ("c",)))
        result = update(make_learner(), logical)
        path = Path(tempfile.mkdtemp(prefix="invar-probabilities-")) / "probabilities.json"
        digest = save(path, LOGICAL_ORDER, result, invocation={"program": "p"}, request={"order": list(LOGICAL_ORDER)})
        document = json.loads(path.read_bytes())
        self.assertEqual(set(document), {"format", "invocation", "request", "samples"})
        self.assertEqual(document["format"], FORMAT)
        self.assertEqual([item["sample"] for item in document["samples"]], list(LOGICAL_ORDER))
        records = logical.exchange.records
        proximal = {record[1]: list(record[2]) for record in records if record[0] == "proximal"}
        currents = [(record[1], record[2], list(record[3])) for record in records if record[0] == "current"]
        for item in document["samples"]:
            name = item["sample"]
            self.assertEqual(item["dtype"], "F32")
            self.assertEqual([(entry["step"], name, entry["current"]) for entry in item["steps"]],
                             [entry for entry in currents if entry[1] == name])
            first = [entry["current"] for entry in item["steps"] if entry["step"] == 0]
            self.assertEqual(item["proximal"], first[0] if first else proximal[name])
        self.assertEqual([entry["step"] for entry in document["samples"][2]["steps"]], [0, 2])
        with self.assertRaises(FileExistsError):
            save(path, LOGICAL_ORDER, result, invocation={}, request={})
        self.assertEqual(len(digest), 64)

    def test_words_preserve_signed_zero_and_do_not_alias_tensors(self):
        for dtype, expected in ((torch.float32, NEGATIVE_ZERO), (torch.float64, 1 << 63)):
            with self.subTest(dtype=dtype):
                value = torch.tensor([-0.0], dtype=dtype)
                observed = words(value)
                value.fill_(1)
                self.assertEqual(observed, (expected,))


if __name__ == "__main__":
    unittest.main()
