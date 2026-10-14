import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

from importlib.metadata import version
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tokenizers import AddedToken, decoders, normalizers

from worker.hf.operation import description, digest, load, verify
from worker.tokenization import PROMPT_SETTINGS, prompt
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

    def test_each_change_moves_only_the_layer_it_belongs_to(self):
        def layers(tokenizer, **settings):
            described = description(tokenizer, **settings)
            return {"token_space": json.dumps(described["token_space"], sort_keys=True), "text": described["text"],
                    "execution": described["execution"]}

        original = layers(make_tokenizer())
        changed = [make_tokenizer() for _ in range(6)]
        changed[0].chat_template += " assistant"
        changed[1].backend_tokenizer.normalizer = normalizers.Lowercase()
        changed[2].clean_up_tokenization_spaces = True
        changed[3].add_tokens([AddedToken("<audio>", special=True)])
        changed[4].backend_tokenizer.decoder = decoders.ByteLevel()
        changed[5].add_tokens([AddedToken("<audio>", special=False)])
        moved = [{name for name, value in layers(tokenizer).items() if value != original[name]} for tokenizer in changed]
        self.assertEqual(moved, [{"text"}, {"text"}, {"text"}, {"token_space", "text"}, {"token_space"}, {"token_space", "text"}])
        self.assertEqual(layers(changed[3])["token_space"], layers(changed[5])["token_space"])
        self.assertNotEqual(layers(changed[3])["text"], layers(changed[5])["text"])
        containers = tuple((name, "np" if name == "return_tensors" else value) for name, value in PROMPT_SETTINGS)
        returned = layers(make_tokenizer(), prompt_settings=containers)
        self.assertEqual({name for name, value in returned.items() if value != original[name]}, {"execution"})
        for package in ("tokenizers", "transformers", "jinja2"):
            with self.subTest(package=package), \
                    patch("worker.hf.operation.version", lambda name, package=package: "0" if name == package else version(name)):
                upgraded = layers(make_tokenizer())
            self.assertEqual({name for name, value in upgraded.items() if value != original[name]}, {"execution"})

    def test_a_description_assigns_every_backend_field(self):
        backend = json.loads(make_tokenizer().backend_tokenizer.to_str())
        gapped = {**backend["model"], "vocab": {**backend["model"]["vocab"], "late": len(backend["model"]["vocab"]) + 1}}
        for value in ({**backend, "extra": None}, {**backend, "model": gapped},
                      {**backend, "added_tokens": [{**backend["added_tokens"][0], "extra": True}]}):
            stub = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda value=value: json.dumps(value)))
            with self.subTest(value=sorted(value)), self.assertRaises(ValueError):
                description(stub)

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
