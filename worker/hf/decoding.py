import torch


def description():
    return {"format": "invar-request-cache-v1", "implementation": "model-returned native cache",
            "prefill": "complete prompt", "decode": "one newly sampled token",
            "attention_mask": "complete logical prefix", "lifetime": "one generation call",
            "reuse_across_requests": False, "training_cache": False}


def forward(model, tokens, *, cache):
    pending = tokens if cache is None else tokens[:, -1:]
    output = model(input_ids=pending, attention_mask=torch.ones_like(tokens),
                   past_key_values=cache, use_cache=True, logits_to_keep=1)
    if output.past_key_values is None:
        raise RuntimeError("Generation did not return the requested native cache")
    return output.logits[:, -1, :], output.past_key_values
