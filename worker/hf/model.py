from pathlib import Path
import re

import torch
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

from worker import core
from worker.cohort import fields
from worker.hf import backend
from worker.hf import decoding
from worker import implementation
from worker.hf.metrics import report

FORMAT = "invar-hf-model-v1"
LORA_RANK = 8
LORA_ALPHA = 16
DEFAULT_SEED = 17
CPU_THREADS = 4


def declared(path):
    value = fields(core.decode(Path(path).read_text()), "format model revision")
    if value["format"] != FORMAT:
        raise ValueError("Expected a Hugging Face model declaration")
    if not isinstance(value["model"], str) or not value["model"]:
        raise ValueError("A model declaration requires a repository name")
    if not isinstance(value["revision"], str) or re.fullmatch(r"[0-9a-f]{40}", value["revision"]) is None:
        raise ValueError("A model declaration requires a pinned commit")
    return value["model"], value["revision"]


def snapshot(cache, identity):
    from huggingface_hub import snapshot_download

    return snapshot_download(identity[0], revision=identity[1], cache_dir=cache, local_files_only=True)


def check_loading(loading, *, emit=report):
    diagnostic = dict(loading)
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        diagnostic[key] = sorted(loading[key])
    emit("loading", diagnostic)
    if any(loading.values()):
        raise RuntimeError("Checkpoint loading did not match the selected architecture")


def load_model(path, *, role, identity, emit=report, adapted=True):
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
    check_loading(loading, emit=emit)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = []
    if adapted:
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
    model.gradient_checkpointing_disable()
    model.eval()
    emit("profile", {"model": identity[0], "revision": identity[1], "targets": targets,
                       "generation": decoding.description(),
                       "numerical": backend.description(), "implementation": implementation.current(role),
                       "quantization": quantization.to_dict(),
                       "attention": "eager", "hub_kernels": False,
                       "gradient_checkpointing": model.is_gradient_checkpointing,
                       "base_preparation": "freeze without bulk dtype promotion",
                       "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
                       "parameters": {name: {"shape": list(p.shape), "dtype": str(p.dtype),
                                             "trainable": p.requires_grad}
                                      for name, p in model.named_parameters()}})
    return model


def adapter_state(model):
    if not isinstance(model, PeftModel):
        return {}
    return {name: value.detach().cpu().contiguous().clone()
            for name, value in get_peft_model_state_dict(model).items()}
