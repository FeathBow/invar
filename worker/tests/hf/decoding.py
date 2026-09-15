import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import unittest
from dataclasses import replace

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from worker.hf.rollout import Request, generate, logprobs
from worker.hf.tensors import assert_equal
from worker.tests.hf.inference import MODEL_SEED, model
from worker.tests.hf.tokenization import WORDS, make_tokenizer

TOKEN_LIMIT = 4
HIDDEN_SIZE = 32
INTERMEDIATE_SIZE = 64
ATTENTION_HEADS = 2
HEAD_SIZE = 16
LORA_RANK = 2
REQUEST = Request(sample="cache", group="cache", prompt="Compute the answer.",
                  seed=MODEL_SEED, limit=TOKEN_LIMIT, temperature=0.8)


def hybrid():
    with torch.random.fork_rng():
        torch.manual_seed(MODEL_SEED)
        config = Qwen3_5Config(
            text_config={"vocab_size": len(WORDS), "hidden_size": HIDDEN_SIZE,
                         "intermediate_size": INTERMEDIATE_SIZE, "num_hidden_layers": 2,
                         "num_attention_heads": ATTENTION_HEADS, "num_key_value_heads": ATTENTION_HEADS,
                         "head_dim": HEAD_SIZE, "linear_key_head_dim": HEAD_SIZE,
                         "linear_value_head_dim": HEAD_SIZE, "linear_num_key_heads": ATTENTION_HEADS,
                         "linear_num_value_heads": ATTENTION_HEADS,
                         "layer_types": ["linear_attention", "full_attention"],
                         "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                             "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]}},
            vision_config={"depth": 1, "hidden_size": HIDDEN_SIZE, "intermediate_size": INTERMEDIATE_SIZE,
                           "num_heads": ATTENTION_HEADS, "out_hidden_size": HIDDEN_SIZE,
                           "patch_size": 2, "num_position_embeddings": 16})
        config.text_config._attn_implementation = "eager"
        base = Qwen3_5ForConditionalGeneration(config)
        targets = [name for name, _ in base.named_modules()
                   if name.startswith("model.language_model.layers.")
                   and name.rsplit(".", 1)[-1] in {"gate_proj", "up_proj", "down_proj"}]
        result = get_peft_model(base, LoraConfig(r=LORA_RANK, lora_alpha=LORA_RANK,
                                                target_modules=targets, task_type="CAUSAL_LM"))
    return result.eval()


def observed_generation(loaded, tokenizer, request):
    calls, caches = [], []

    def before(module, args, kwargs):
        calls.append((kwargs["input_ids"].clone(), kwargs["attention_mask"].clone(),
                      kwargs["past_key_values"], kwargs["use_cache"]))

    def after(module, args, output):
        caches.append(output.past_key_values)

    pre = loaded.register_forward_pre_hook(before, with_kwargs=True)
    post = loaded.register_forward_hook(after)
    try:
        result = generate(loaded, tokenizer, request, device="cpu")
    finally:
        pre.remove()
        post.remove()
    return result, calls, caches


class DecodingTests(unittest.TestCase):
    def test_native_prefill_then_single_tokens_with_complete_mask_and_own_cache(self):
        for constructor in (model, hybrid):
            with self.subTest(model=constructor.__name__):
                result, calls, caches = observed_generation(constructor(), make_tokenizer(), REQUEST)
                self.assertGreater(len(calls), 1)
                for index, (tokens, mask, incoming, enabled) in enumerate(calls):
                    length = result.prompt_length + index
                    expected = result.tokens[:, :length] if index == 0 else result.tokens[:, length - 1:length]
                    assert_equal(tokens, expected)
                    assert_equal(mask, torch.ones_like(result.tokens[:, :length]))
                    self.assertTrue(enabled)
                    self.assertIs(incoming, None if index == 0 else caches[index - 1])
                self.assertIsNotNone(caches[-1])

    def test_requests_reset_kv_and_recurrent_state_and_preserve_global_rng(self):
        loaded, tokenizer = hybrid(), make_tokenizer()
        rng = torch.get_rng_state()
        first, _, first_cache = observed_generation(loaded, tokenizer, REQUEST)
        generate(loaded, tokenizer, replace(REQUEST, prompt="#### 437", seed=MODEL_SEED + 1), device="cpu")
        second, calls, second_cache = observed_generation(loaded, tokenizer, REQUEST)
        self.assertIsNone(calls[0][2])
        self.assertIsNot(first_cache[0], second_cache[0])
        assert_equal(first.tokens, second.tokens)
        assert_equal(first.behavior, second.behavior)
        assert_equal(rng, torch.get_rng_state())
        self.assertEqual(first.text, second.text)
        self.assertEqual(first.truncated, second.truncated)
        recurrent = first_cache[-1].layers[0]
        self.assertTrue(all(recurrent.is_conv_states_initialized.values()))
        self.assertTrue(all(recurrent.is_recurrent_states_initialized.values()))
        self.assertTrue(all(value.isfinite().all() for value in recurrent.conv_states.values()))
        self.assertTrue(all(value.isfinite().all() for value in recurrent.recurrent_states.values()))
        self.assertTrue(first_cache[-1].layers[1].is_initialized)

    def test_training_after_cached_generation_has_fresh_autograd_and_no_cache(self):
        loaded, tokenizer = hybrid(), make_tokenizer()
        trajectory = generate(loaded, tokenizer, REQUEST, device="cpu")
        loaded.train()
        observed = []

        def before(module, args, kwargs):
            observed.append(kwargs)

        hook = loaded.register_forward_pre_hook(before, with_kwargs=True)
        try:
            values = logprobs(loaded, trajectory, device="cpu")
            values.sum().backward()
        finally:
            hook.remove()
        self.assertEqual(len(observed), 1)
        self.assertFalse(observed[0]["use_cache"])
        self.assertNotIn("past_key_values", observed[0])
        gradients = [value.grad for value in loaded.parameters() if value.requires_grad]
        self.assertTrue(all(value is not None and value.isfinite().all() for value in gradients))
        self.assertTrue(any(torch.count_nonzero(value).item() for value in gradients))

    def test_missing_native_cache_fails_without_recomputing_the_prefix(self):
        loaded = model()

        def omit_cache(module, args, output):
            output.past_key_values = None

        hook = loaded.register_forward_hook(omit_cache)
        try:
            with self.assertRaisesRegex(RuntimeError, "did not return.*native cache"):
                generate(loaded, make_tokenizer(), REQUEST, device="cpu")
        finally:
            hook.remove()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
