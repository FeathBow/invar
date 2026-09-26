import unittest

try:
    import mlx  # noqa: F401
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import math
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import Model, ModelArgs
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from worker.mlx import adapter as mlx_adapter
from worker.mlx import numerics as mlx_numerics
from worker.mlx import probability as mlx_probability
from worker.mlx import rollout as mlx_rollout
from worker.mlx import tensors as mlx_tensors
from worker.mlx import tokenization as mlx_tokenization
from worker import scalar
from worker.trajectory import Request
from worker.implementation import INFERENCE

WORDS = ("[UNK]", "[EOS]", "one", "two", "three", "four", "five", "assistant")
LAYERS = 4


def tokenizer(*, vocabulary=len(WORDS)):
    if vocabulary < len(WORDS):
        raise ValueError("The fixture vocabulary must retain its prompt and special tokens")
    words = (*WORDS, *(f"token{index}" for index in range(len(WORDS), vocabulary)))
    backend = Tokenizer(models.WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    result = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")
    result.chat_template = "{{ messages[0]['content'] }}{% if add_generation_prompt %} assistant{% endif %}"
    return result


def model(*, uniform_head=True, vocabulary=len(WORDS)):
    config = {"model_type": "qwen3_5_text", "hidden_size": 64, "intermediate_size": 128,
              "num_hidden_layers": LAYERS, "num_attention_heads": 4, "head_dim": 16,
              "num_key_value_heads": 2, "vocab_size": vocabulary, "linear_num_value_heads": 4,
              "linear_num_key_heads": 2, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
              "full_attention_interval": 4, "partial_rotary_factor": 0.5}
    value = Model(ModelArgs(model_type="qwen3_5", text_config=config))
    value.set_dtype(mx.bfloat16)
    if uniform_head:
        value.language_model.lm_head.weight = mx.zeros_like(value.language_model.lm_head.weight)
    nn.quantize(value, group_size=64, bits=4, mode="affine")
    mlx_adapter.create(value, layers=LAYERS)
    mx.eval(value.parameters())
    return value, config


class RolloutTests(unittest.TestCase):
    def test_actual_native_batch_preserves_behavior_and_learner_rng(self):
        numerical, config = model()
        mlx_numerics.PRIMARY.install(numerical)
        text = tokenizer()
        requests = tuple(Request(sample=str(index), group="group", prompt=prompt,
                                  seed=17 + index, limit=limit, temperature=0.8)
                         for index, (prompt, limit) in enumerate((("one", 2), ("one two three", 4), ("two", 3))))
        before = {"key": mx.array(mx.random.state[0])}
        identity = mlx_adapter.images(numerical, config, numerics=mlx_numerics.PRIMARY.observe(numerical, INFERENCE))
        trajectories = mlx_rollout.generate(numerical, text, requests, sampling=mlx_rollout.Sampling(batch_size=2, prefill_step=2))
        self.assertEqual(tuple(value.request for value in trajectories), requests)
        expected = scalar.word(math.log(1 / len(WORDS)))
        for value in trajectories:
            count = value.tokens.shape[-1] - value.prompt_length
            self.assertEqual(mlx_probability.words(value.behavior), (expected,) * count)
            self.assertGreater(count, 0)
            self.assertLessEqual(count, value.request.limit)
            prefix = mlx_tokenization.prompt(text, value.request.prompt)
            self.assertEqual(value.tokens[0, :value.prompt_length].tolist(), prefix[0].tolist())
            self.assertEqual(value.truncated, value.tokens[0, -1].item() != text.eos_token_id)
        self.assertTrue(mlx_tensors.equal(before, {"key": mx.random.state[0]}))
        self.assertEqual(mlx_adapter.images(numerical, config, numerics=mlx_numerics.PRIMARY.observe(numerical, INFERENCE)), identity)


if __name__ == "__main__":
    unittest.main()
