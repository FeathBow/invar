import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import tempfile
import unittest
from pathlib import Path

from tokenizers import normalizers

from worker.hf.operation import digest, load, verify
from worker.tokenization import decode, prompt
from worker.tests.hf.tokenization import make_tokenizer


class TextIdentityTests(unittest.TestCase):
    def test_saved_tokenizer_and_prompt_use_preserve_identity(self):
        tokenizer = make_tokenizer()
        expected = digest(tokenizer)
        path = Path(tempfile.mkdtemp(prefix="invar-text-identity-"))
        tokenizer.save_pretrained(path)
        restored = load(path)
        self.assertEqual(expected, verify(restored, expected))
        prompt(restored, "Compute the answer.")
        self.assertEqual(expected, digest(restored))

    def test_changed_backend_template_eos_and_decode_settings_are_rejected(self):
        original = digest(make_tokenizer())
        tokenizers = [make_tokenizer() for _ in range(4)]
        tokenizers[0].backend_tokenizer.normalizer = normalizers.Lowercase()
        tokenizers[1].chat_template += " assistant"
        tokenizers[2].eos_token = "437"
        tokenizers[3].clean_up_tokenization_spaces = True
        for tokenizer in tokenizers:
            with self.subTest(tokenizer=digest(tokenizer)):
                with self.assertRaisesRegex(ValueError, "requested tokenizer identity"):
                    verify(tokenizer, original)

    def test_effective_cleanup_change_changes_actual_decoding(self):
        tokenizer = make_tokenizer()
        expected = digest(tokenizer)
        observed = decode(tokenizer, [7, 5])
        tokenizer.clean_up_tokenization_spaces = True
        self.assertNotEqual(observed, decode(tokenizer, [7, 5]))
        self.assertNotEqual(expected, digest(tokenizer))

    def test_fixed_prompt_settings_override_transient_backend_limits(self):
        tokenizer = make_tokenizer()
        expected = digest(tokenizer)
        baseline = prompt(tokenizer, "Compute the answer.").tolist()
        tokenizer.backend_tokenizer.enable_truncation(max_length=1)
        tokenizer.backend_tokenizer.enable_padding(length=12, pad_id=0, pad_token="[UNK]")
        self.assertEqual(expected, digest(tokenizer))
        self.assertEqual(baseline, prompt(tokenizer, "Compute the answer.").tolist())
        self.assertEqual(expected, digest(tokenizer))


if __name__ == "__main__":
    unittest.main()
