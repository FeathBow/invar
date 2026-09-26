import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import copy
import unittest
from dataclasses import replace

import torch

from worker.tests.hf.learning import make_learner
from worker.hf.tensors import assert_equal
from worker.hf.step import file_digest, restore_inputs
from worker.tests.hf.step import prepared
from worker.hf.operation import digest
from worker.tokenization import validate


class TextCheckpointTests(unittest.TestCase):
    def rejected(self, request, options, message):
        consumer = make_learner()
        model = copy.deepcopy(consumer.model.state_dict())
        optimizer = copy.deepcopy(consumer.optimizer.state_dict())
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        with self.assertRaisesRegex(ValueError, message):
            restore_inputs(consumer.model, request, options, tokenizer=options.tokenizer)
        assert_equal(model, consumer.model.state_dict())
        assert_equal(optimizer, consumer.optimizer.state_dict())
        assert_equal(cpu_rng, torch.get_rng_state())
        assert_equal(cuda_rng, torch.cuda.get_rng_state_all())

    def test_unused_vocabulary_change_cannot_consume_the_original_update(self):
        _, _, request, options = prepared()
        options.tokenizer.add_tokens(["unused_added_word"])
        validate(options.tokenizer, request.samples)
        self.assertNotEqual(digest(options.tokenizer), request.tokenizer)
        self.rejected(request, options, "requested tokenizer identity")

    def test_matching_new_request_cannot_reinterpret_the_saved_learner(self):
        _, _, request, options = prepared()
        saved = torch.load(options.checkpoint / "learner.pt", weights_only=True)
        self.assertEqual(saved["tokenizer"], request.tokenizer)
        options.tokenizer.add_tokens(["unused_added_word"])
        validate(options.tokenizer, request.samples)
        changed = replace(request, tokenizer=digest(options.tokenizer))
        self.rejected(changed, options, "requested tokenizer identity")

    def test_missing_checkpoint_identity_is_not_inferred_from_current_request(self):
        _, _, request, options = prepared()
        path = options.checkpoint / "learner.pt"
        saved = torch.load(path, weights_only=True)
        torch.save({key: value for key, value in saved.items() if key != "tokenizer"}, path)
        changed = replace(request, learner=file_digest(path))
        self.rejected(changed, options, "lowercase SHA-256 identity")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
