import argparse
import importlib.metadata
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

import assembly
import backend
import implementation
import binding
import frozen
import operation
from metrics import measure, report, update_profile
from tensors import assert_equal, digest

REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MODEL = "Qwen/Qwen3.8-27B"
IGNORE_LABEL = -100
LORA_RANK = 8
LORA_ALPHA = 16
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
TEMPERATURE = 0.8
DEFAULT_TOKENS = 32
DEFAULT_SEED = 17
CPU_THREADS = 4
PACKAGES = ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "huggingface-hub")


@dataclass(frozen=True, kw_only=True)
class CheckpointIdentity:
    adapter: str
    base: str
    assembly: str
    tokenizer: str


def check_loading(loading):
    diagnostic = dict(loading)
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        diagnostic[key] = sorted(loading[key])
    report("loading", diagnostic)
    if any(loading.values()):
        raise RuntimeError("Checkpoint loading did not match the selected architecture")


def load_model(path):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=quantization, attn_implementation="eager",
        use_kernels=False, output_loading_info=True,
    )
    check_loading(loading)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = [name for name, _ in model.named_modules()
               if name.startswith("model.language_model.layers.")
               and name.rsplit(".", 1)[-1] in {"gate_proj", "up_proj", "down_proj"}]
    if not targets:
        raise RuntimeError("No language-model adapter targets resolved")
    config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=targets,
                        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    if any(p.requires_grad and p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("Adapter parameters do not match the FP32 training profile")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.eval()
    report("profile", {"model": MODEL, "revision": REVISION, "targets": targets,
                       "numerical": backend.description(), "implementation": implementation.current(),
                       "quantization": quantization.to_dict(),
                       "attention": "eager", "hub_kernels": False,
                       "base_preparation": "freeze without bulk dtype promotion",
                       "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
                       "parameters": {name: {"shape": list(p.shape), "dtype": str(p.dtype),
                                             "trainable": p.requires_grad}
                                      for name, p in model.named_parameters()}})
    return model


def forward(model, tokens):
    return model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                 use_cache=False, logits_to_keep=1).logits[:, -1, :]


def sample(model, tokenizer, options):
    messages = [{"role": "user", "content": options.prompt}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False, return_tensors="pt", return_dict=True)
    tokens = inputs["input_ids"].to("cuda")
    prompt_length = tokens.shape[1]
    generator = torch.Generator(device="cuda").manual_seed(options.seed)
    observations = []
    with torch.no_grad():
        for _ in range(options.tokens):
            probabilities = torch.softmax(forward(model, tokens).float() / TEMPERATURE, dim=-1)
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("Non-finite sampling distribution")
            chosen = torch.multinomial(probabilities, 1, generator=generator)
            probability = probabilities.gather(-1, chosen) / probabilities.sum(-1, keepdim=True)
            observations.append({"token": chosen.item(), "logprob": probability.log().item()})
            tokens = torch.cat((tokens, chosen), dim=1)
            if chosen.item() == tokenizer.eos_token_id:
                break
    report("sample", {"seed": options.seed, "temperature": TEMPERATURE,
                      "prompt_tokens": tokens[0, :prompt_length].tolist(),
                      "observations": observations,
                      "text": tokenizer.decode(tokens[0, prompt_length:], skip_special_tokens=False)})
    return tokens, prompt_length


def adapter_state(model):
    return {name: value.detach().cpu().contiguous().clone()
            for name, value in get_peft_model_state_dict(model).items()}


def loss_for(model, tokens, labels):
    output = model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                   use_cache=False, labels=labels)
    return output.loss, str(output.logits.dtype)


def checked_gradient_norm(gradients):
    if not gradients or any(g is None or not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Missing or non-finite adapter gradients")
    norm = torch.stack([g.double().square().sum() for g in gradients]).sum().sqrt().item()
    if norm == 0:
        raise RuntimeError("The selected loss produced a zero adapter gradient")
    return norm


def update(model, optimizer, batch):
    tokens, prompt_length = batch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    labels = tokens.clone()
    labels[:, :prompt_length] = IGNORE_LABEL
    loss, logits_dtype = measure("loss_forward", lambda: loss_for(model, tokens, labels), memory=False)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss")
    measure("backward", loss.backward, memory=False)
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    norm = checked_gradient_norm(gradients)
    measure("optimizer", optimizer.step, memory=False)
    if any(not torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
        raise RuntimeError("Non-finite adapter after optimizer step")
    model.eval()
    report("update", {"loss": loss.item(), "gradient_norm": norm,
                      "objective": "response-token cross entropy; feasibility only, not RL",
                      "dtypes": {"loss": str(loss.dtype), "logits": logits_dtype,
                                 **update_profile(gradients, optimizer)},
                      "optimizer_steps": [state["step"].item() for state in optimizer.state.values()]})


def checkpoint(model, optimizer, output, *, tokenizer, expected):
    parameters = binding.parameters(model, optimizer)
    tokenizer_digest = operation.digest(tokenizer)
    configuration = assembly.digest(model)
    base = frozen.digest(model)
    state = adapter_state(model)
    actual = CheckpointIdentity(adapter=digest(state), base=base, assembly=configuration,
                                tokenizer=tokenizer_digest)
    if expected is not None and actual != expected:
        raise RuntimeError("Checkpoint identity differs from the declared successor")
    save_file(state, output / "adapter.safetensors")
    training = {**asdict(actual),
                "parameters": parameters,
                "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}
    torch.save(training, output / "learner.pt")
    assert_equal(state, load_file(output / "adapter.safetensors"))
    assert_equal(training, torch.load(output / "learner.pt", weights_only=True))
    return state


def restore(model, optimizer, output, *, tokenizer):
    state = load_file(output / "adapter.safetensors")
    training = torch.load(output / "learner.pt", weights_only=True)
    restore_state(model, optimizer, (state, training), tokenizer=tokenizer)


def restore_state(model, optimizer, snapshot, *, tokenizer):
    state, training = snapshot
    if training["adapter"] != digest(state):
        raise RuntimeError("Checkpoint adapter binding mismatch")
    operation.verify(tokenizer, training.get("tokenizer"))
    assembly.verify(model, training.get("assembly"))
    frozen.verify(model, training.get("base"))
    training = binding.restore(training, model, optimizer)
    set_peft_model_state_dict(model, state)
    assert_equal(state, adapter_state(model))
    optimizer.load_state_dict(training["optimizer"])
    torch.set_rng_state(training["cpu_rng"])
    torch.cuda.set_rng_state_all(training["cuda_rng"])
    assert_equal(training["optimizer"], optimizer.state_dict())


def learning(model, batch, output, *, tokenizer):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, betas=ADAM_BETAS,
                                  eps=ADAM_EPSILON, weight_decay=0.0, foreach=False, fused=False)
    before = digest(adapter_state(model))
    measure("first_update", lambda: update(model, optimizer, batch))
    published = checkpoint(model, optimizer, output, tokenizer=tokenizer, expected=None)
    if digest(published) == before:
        raise RuntimeError("Optimizer step did not change the adapter")
    measure("uninterrupted_update", lambda: update(model, optimizer, batch))
    uninterrupted = adapter_state(model)
    torch.save(optimizer.state_dict(), output / "continuation.pt")
    measure("restore", lambda: restore(model, optimizer, output, tokenizer=tokenizer))
    measure("resumed_update", lambda: update(model, optimizer, batch))
    resumed = adapter_state(model)
    assert_equal(uninterrupted, resumed)
    assert_equal(torch.load(output / "continuation.pt", weights_only=True), optimizer.state_dict())
    report("continuation", {"initial": before, "published": digest(published),
                            "continued": digest(uninterrupted), "resumed": digest(resumed),
                            "resumed_bitwise_equal": True})


def arguments():
    parser = argparse.ArgumentParser(description="Real-model execution feasibility; not Invar certification")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="What is 19 multiplied by 23? Explain briefly.")
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    options = parser.parse_args()
    if options.tokens <= 0:
        parser.error("--tokens must be positive")
    if not options.cache.is_dir():
        parser.error("--cache must contain the downloaded pinned checkpoint")
    return options


def main():
    options = arguments()
    path = snapshot_download(MODEL, revision=REVISION, cache_dir=options.cache, local_files_only=True)
    options.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(CPU_THREADS)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)
    report("environment", {"versions": {name: importlib.metadata.version(name) for name in PACKAGES},
                           "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
                           "float32_matmul_precision": torch.get_float32_matmul_precision(),
                           "cudnn_tf32": torch.backends.cudnn.allow_tf32})
    model = measure("load", lambda: load_model(path))
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    batch = measure("generation", lambda: sample(model, tokenizer, options))
    learning(model, batch, options.output, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
