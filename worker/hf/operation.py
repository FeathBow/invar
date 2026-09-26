import argparse
import hashlib
import json
from pathlib import Path

from worker.cohort import identity
from worker.tokenization import DECODE_SETTINGS, PROMPT_SETTINGS

FORMAT = "invar-tokenizer-operation-v1"
CLEANUP_FIELDS = ("clean_up_tokenization_spaces",
                  "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output")


def description(tokenizer, *, prompt_settings=PROMPT_SETTINGS):
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    fixed = {name: value for name, value in backend.items() if name not in ("padding", "truncation")}
    return {"format": FORMAT, "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "backend": fixed, "template": tokenizer.get_chat_template(),
            "special_tokens": tokenizer.special_tokens_map, "eos": tokenizer.eos_token_id,
            "split_special_tokens": tokenizer.split_special_tokens,
            "cleanup": {name: getattr(tokenizer, name) for name in CLEANUP_FIELDS},
            "prompt": dict(prompt_settings), "decode": dict(DECODE_SETTINGS)}


def digest(tokenizer, *, prompt_settings=PROMPT_SETTINGS):
    encoded = json.dumps(description(tokenizer, prompt_settings=prompt_settings), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify(tokenizer, expected, *, prompt_settings=PROMPT_SETTINGS):
    identity(expected)
    actual = digest(tokenizer, prompt_settings=prompt_settings)
    if actual != expected:
        raise ValueError("Loaded tokenizer differs from the requested tokenizer identity")
    return actual


def load(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def main():
    from huggingface_hub import snapshot_download
    from worker.hf.model import MODEL, REVISION

    parser = argparse.ArgumentParser(description="Identify the pinned tokenizer operation without loading model weights")
    parser.add_argument("--cache", type=Path, required=True)
    options = parser.parse_args()
    path = snapshot_download(MODEL, revision=REVISION, cache_dir=options.cache, local_files_only=True)
    print(digest(load(path)))


if __name__ == "__main__":
    main()
