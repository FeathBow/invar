from dataclasses import replace
import unittest

from worker.tests.mlx.crossscore import SAMPLING, loaded, path, requests

import mlx.core as mx

from worker.mlx.crossscore import PathSampler, coordinates
from worker.mlx.probability import words
from worker.mlx.rollout import generate, run


class RecordedSampler(PathSampler):
    def __init__(self, request, selected, *, width):
        super().__init__(request, selected, width=width)
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
        cls.width = coordinates(cls.runtime.model)["output"]

    def test_actual_engine_context_mismatch_stops_before_sampling(self):
        request, selected = self.requests[0], self.paths[0]
        wrong = replace(selected, prefix=((selected.prefix[0] + 1) % len(self.runtime.tokenizer), *selected.prefix[1:]))
        sampler = PathSampler(request, wrong, width=self.width)
        with self.assertRaisesRegex(ValueError, "context"):
            run(self.runtime.model, (selected.prefix,), limits=(len(selected.response),), stop=(), sampling=SAMPLING, samplers=(sampler,))
        self.assertEqual(sampler.position, 0)
        self.assertFalse(sampler.pending)

    def test_native_contexts_cover_each_request_and_lookahead_without_word_changes(self):
        samplers = tuple(RecordedSampler(request, selected, width=self.width)
                         for request, selected in zip(self.requests, self.paths, strict=True))
        scored = run(self.runtime.model, tuple(selected.prefix for selected in self.paths),
                     limits=tuple(len(selected.response) for selected in self.paths), stop=(), sampling=SAMPLING, samplers=samplers)
        for sampler, (_, behavior, _), original in zip(samplers, scored, self.generated, strict=True):
            with self.subTest(request=sampler.path.prefix):
                selected = sampler.path
                self.assertEqual(sampler.contexts, [(*selected.prefix, *selected.response[:step])
                                                   for step in range(len(selected.response) + 1)])
                self.assertEqual(words(mx.stack(behavior)), words(original.behavior))
                sampler.completed()


if __name__ == "__main__":
    unittest.main()
