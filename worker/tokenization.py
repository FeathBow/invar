PROMPT_SETTINGS = (("tokenize", True), ("add_generation_prompt", True), ("enable_thinking", False),
                   ("padding", False), ("truncation", False), ("return_tensors", "pt"), ("return_dict", True))
DECODE_SETTINGS = (("skip_special_tokens", True),)


def prompt(tokenizer, text):
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": text}], **dict(PROMPT_SETTINGS))
    return inputs["input_ids"]


def decode(tokenizer, tokens):
    return tokenizer.decode(tokens, **dict(DECODE_SETTINGS))


def validate(tokenizer, samples, *, encode=prompt):
    for item in samples:
        prefix = tuple(encode(tokenizer, item.prompt)[0].tolist())
        if prefix != item.tokens[:item.prompt_length]:
            raise ValueError(f"Observed prompt tokens differ from the loaded tokenizer: {item.sample}")
        response = item.tokens[item.prompt_length:]
        if tokenizer.eos_token_id in response[:-1]:
            raise ValueError(f"Observed response continues after EOS: {item.sample}")
        ended = response[-1] == tokenizer.eos_token_id
        if item.truncated != (not ended):
            raise ValueError(f"Observed truncation differs from the loaded tokenizer: {item.sample}")
        text = decode(tokenizer, response)
        if text != item.text:
            raise ValueError(f"Observed text differs from the loaded tokenizer: {item.sample}")
