import mlx.core as mx

from worker.hf import operation
from worker import tokenization

PROMPT_SETTINGS = tuple((key, value) for key, value in tokenization.PROMPT_SETTINGS
                        if key not in ("return_tensors", "return_dict")) + (("return_dict", False),)


def prompt(tokenizer, text):
    values = tokenizer.apply_chat_template([{"role": "user", "content": text}], **dict(PROMPT_SETTINGS))
    return mx.array([values], dtype=mx.int32)


def digest(tokenizer):
    return operation.digest(tokenizer, prompt_settings=PROMPT_SETTINGS)


def verify(tokenizer, expected):
    return operation.verify(tokenizer, expected, prompt_settings=PROMPT_SETTINGS)


def validate(tokenizer, samples):
    tokenization.validate(tokenizer, samples, encode=prompt)
