from dataclasses import replace
import unittest

from worker.tests.mlx.crossscore import SAMPLING, loaded, path, requests

from worker.mlx.crossscore import PathSampler
from worker.mlx.probability import words
from worker.mlx.rollout import execute, generate


class RecordedSampler(PathSampler):
    def __init__(self, request, selected):
        super().__init__(request, selected)
        self.contexts = []

    def process_logits(self, tokens, logits):
        self.contexts.append(tuple(tokens.tolist()))
        return super().process_logits(tokens, logits)


class NativeContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = loaded(71)
        cls.requests = requests()
        cls.generated = generate(cls.runtime.model, cls.runtime.tokenizer, cls.requests, sampling=SAMPLING)
        cls.paths = tuple(path(value) for value in cls.generated)

    def test_actual_engine_context_mismatch_stops_before_sampling(self):
        request, selected = self.requests[0], self.paths[0]
        wrong = replace(selected, prefix=((selected.prefix[0] + 1) % len(self.runtime.tokenizer), *selected.prefix[1:]))
        sampler = PathSampler(request, wrong)
        with self.assertRaisesRegex(ValueError, "context"):
            execute(self.runtime.model, self.runtime.tokenizer, (request,), sampling=SAMPLING, samplers=(sampler,))
        self.assertEqual(sampler.position, 0)
        self.assertFalse(sampler.pending)

    def test_native_contexts_cover_each_request_and_lookahead_without_word_changes(self):
        samplers = tuple(RecordedSampler(request, selected) for request, selected in zip(self.requests, self.paths, strict=True))
        scored = execute(self.runtime.model, self.runtime.tokenizer, self.requests, sampling=SAMPLING, samplers=samplers)
        for sampler, actual, original in zip(samplers, scored, self.generated, strict=True):
            with self.subTest(request=sampler.path.prefix):
                selected = sampler.path
                self.assertEqual(sampler.contexts, [(*selected.prefix, *selected.response[:step])
                                                   for step in range(len(selected.response) + 1)])
                self.assertEqual(words(actual.behavior), words(original.behavior))
                sampler.completed()


if __name__ == "__main__":
    unittest.main()
