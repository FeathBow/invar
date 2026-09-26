PROMPT_SETTINGS = (("tokenize", True), ("add_generation_prompt", True), ("enable_thinking", False),
                   ("padding", False), ("truncation", False), ("return_tensors", "pt"), ("return_dict", True))
DECODE_SETTINGS = (("skip_special_tokens", True),)


def prompt(tokenizer, text):
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": text}], **dict(PROMPT_SETTINGS))
    return inputs["input_ids"]


def decode(tokenizer, tokens):
    return tokenizer.decode(tokens, **dict(DECODE_SETTINGS))

