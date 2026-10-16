import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from worker.cohort import identity
from worker.tokenization import DECODE_SETTINGS, PROMPT_SETTINGS

FORMAT = "invar-tokenizer-v2"
SPACE_FORMAT = "invar-token-space-v1"
TEXT_FORMAT = "invar-tokenizer-text-v1"
EXECUTION_FORMAT = "invar-tokenizer-execution-v1"
CLEANUP_FIELDS = ("clean_up_tokenization_spaces",
                  "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output")
CONTAINERS = ("return_tensors", "return_dict")
# The packages that run the backend, the tokenizer class and its chat template.
IMPLEMENTATION = ("tokenizers", "transformers", "jinja2")
# The backend's padding and truncation are call state; the prompt settings declare the calls.
BACKEND_FIELDS = {"version", "truncation", "padding", "added_tokens", "normalizer", "pre_tokenizer",
                  "post_processor", "decoder", "model"}
ADDED_FIELDS = {"id", "content", "single_word", "lstrip", "rstrip", "normalized", "special"}


def canonical(value):
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if type(value) is int:
        return str(value)
    if isinstance(value, str):
        return '"' + "".join('\\"' if character == '"' else "\\\\" if character == "\\"
                             else f"\\u{ord(character):04x}" if ord(character) < 0x20 else character
                             for character in value) + '"'
    if isinstance(value, list):
        return "[" + ",".join(canonical(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(canonical(key) + ":" + canonical(item)
                              for key, item in sorted(value.items(), key=lambda pair: pair[0].encode())) + "}"
    raise ValueError(f"No canonical encoding for {type(value).__name__}")


def sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def description(tokenizer, *, prompt_settings=PROMPT_SETTINGS):
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    if set(backend) != BACKEND_FIELDS:
        raise ValueError("Tokenizer backend has fields this description does not assign")
    model = backend["model"]
    vocabulary = model.get("vocab")
    if not isinstance(vocabulary, dict) or sorted(vocabulary.values()) != list(range(len(vocabulary))):
        raise ValueError("Tokenizer vocabulary must map tokens to the identifiers 0..n-1")
    added = sorted(backend["added_tokens"], key=lambda token: token["id"])
    if any(set(token) != ADDED_FIELDS for token in added):
        raise ValueError("Added tokens have fields this description does not assign")
    settings = dict(prompt_settings)
    space = {"format": SPACE_FORMAT,
             "vocabulary": {"size": len(vocabulary), "sha256": sha256(sorted(vocabulary, key=vocabulary.get))},
             "added": [{"id": token["id"], "content": token["content"]} for token in added],
             "decoder": sha256(backend["decoder"])}
    text = {"format": TEXT_FORMAT,
            "model": {name: value for name, value in model.items() if name != "vocab"},
            "normalizer": backend["normalizer"], "pre_tokenizer": backend["pre_tokenizer"],
            "post_processor": backend["post_processor"],
            "added": [{name: token[name] for name in ADDED_FIELDS - {"content"}} for token in added],
            "template": tokenizer.get_chat_template(), "special_tokens": tokenizer.special_tokens_map,
            "eos": tokenizer.eos_token_id, "split_special_tokens": tokenizer.split_special_tokens,
            "cleanup": {name: getattr(tokenizer, name) for name in CLEANUP_FIELDS},
            "prompt": {name: value for name, value in settings.items() if name not in CONTAINERS},
            "decode": dict(DECODE_SETTINGS)}
    execution = {"format": EXECUTION_FORMAT, "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
                 "serialization": backend["version"], "implementation": {name: version(name) for name in IMPLEMENTATION},
                 "containers": {name: value for name, value in settings.items() if name in CONTAINERS}}
    return {"format": FORMAT, "token_space": space, "text": sha256(text), "execution": sha256(execution)}


def digest(tokenizer, *, prompt_settings=PROMPT_SETTINGS):
    return identified(description(tokenizer, prompt_settings=prompt_settings))


def identified(described):
    return hashlib.sha256(canonical(described).encode()).hexdigest()


def described(tokenizer, expected, *, prompt_settings=PROMPT_SETTINGS):
    identity(expected)
    actual = description(tokenizer, prompt_settings=prompt_settings)
    if identified(actual) != expected:
        raise ValueError("Loaded tokenizer differs from the requested tokenizer identity")
    return actual


def verify(tokenizer, expected, *, prompt_settings=PROMPT_SETTINGS):
    described(tokenizer, expected, prompt_settings=prompt_settings)
    return expected


def load(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def main():
    from worker.hf.model import declared, snapshot

    parser = argparse.ArgumentParser(description="Identify the declared model's tokenizer operation without loading model weights")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    options = parser.parse_args()
    print(digest(load(snapshot(options.cache, declared(options.config)))))


if __name__ == "__main__":
    main()
