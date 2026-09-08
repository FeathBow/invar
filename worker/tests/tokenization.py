import unittest

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from cohort import decode
from .cohort import request
from tokenization import prompt, validate
from operation import digest

WORDS = ("[UNK]", "[EOS]", "Compute", "the", "answer", ".", "####", "437", "438", "assistant", "think")
TEMPLATE = ("{{ messages[0]['content'] }}{% if add_generation_prompt %} assistant{% endif %}"
            "{% if enable_thinking %} think{% endif %}")


def make_tokenizer():
    backend = Tokenizer(models.WordLevel({word: index for index, word in enumerate(WORDS)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.chat_template = TEMPLATE
    return tokenizer


def admitted_request(tokenizer):
    declared = request()
    samples = []
    for item in declared["samples"]:
        prefix = prompt(tokenizer, item["prompt"])[0].tolist()
        response = tokenizer.encode(item["text"], add_special_tokens=False) + [tokenizer.eos_token_id]
        samples.append({**item, "tokens": prefix + response, "prompt_length": len(prefix),
                        "limit": len(response), "behavior_bits": item["behavior_bits"] * len(response)})
    return {**declared, "tokenizer": digest(tokenizer), "samples": samples}


class TokenizationTests(unittest.TestCase):
    def test_real_tokenizer_admits_completed_and_truncated_observations(self):
        tokenizer = make_tokenizer()
        value = admitted_request(tokenizer)
        completed = value["samples"][0]
        truncated = value["samples"][1]
        value["samples"] = [{**completed, "limit": completed["limit"] + 1},
                            {**truncated, "tokens": truncated["tokens"][:-1], "truncated": True,
                             "behavior_bits": truncated["behavior_bits"][:-1], "limit": truncated["limit"] - 1}]
        parsed = decode(value)
        rng = torch.get_rng_state()
        validate(tokenizer, parsed.samples)
        self.assertEqual(prompt(tokenizer, completed["prompt"])[0].tolist(), [2, 3, 4, 5, 9])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(parsed, decode(value))

    def test_lexically_valid_observations_must_match_the_loaded_tokenizer(self):
        tokenizer = make_tokenizer()
        value = admitted_request(tokenizer)
        original = value["samples"][0]
        boundary = original["prompt_length"]
        changes = [{"tokens": [0, *original["tokens"][1:]]},
                   {"text": "#### 438"}, {"truncated": True},
                   {"tokens": [*original["tokens"][:-1], 7]},
                   {"tokens": [*original["tokens"][:boundary], tokenizer.eos_token_id,
                               *original["tokens"][boundary + 1:]]}]
        for changed in changes:
            with self.subTest(changed=changed):
                parsed = decode({**value, "samples": [{**original, **changed}, value["samples"][1]]})
                with self.assertRaisesRegex(ValueError, "loaded tokenizer|after EOS"):
                    validate(tokenizer, parsed.samples)

    def test_chat_template_and_eos_changes_are_observable(self):
        tokenizer = make_tokenizer()
        parsed = decode(admitted_request(tokenizer))
        tokenizer.chat_template = "{{ messages[0]['content'] }}"
        with self.assertRaisesRegex(ValueError, "prompt tokens"):
            validate(tokenizer, parsed.samples)
        tokenizer.chat_template = TEMPLATE
        tokenizer.eos_token = "437"
        with self.assertRaisesRegex(ValueError, "after EOS"):
            validate(tokenizer, parsed.samples)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
