# vLLM patches

The CUDA worker runs vLLM 0.28.0 with the BitsAndBytes plugin. [The source pins](vllm/sources.json) record both upstream repositories and the exact revisions these patches target. Four patches in [vllm/](vllm) change these sources.

| Patch | Tree | Files |
| --- | --- | --- |
| [fp32.patch](vllm/fp32.patch) | vLLM | `vllm/config/lora.py`, `vllm/lora/ops/triton_ops/kernel_utils.py`, `lora_shrink_op.py`, `lora_expand_op.py` |
| [fp32padding.patch](vllm/fp32padding.patch) | vLLM | `vllm/lora/ops/triton_ops/lora_shrink_op.py` |
| [bnb.patch](vllm/bnb.patch) | plugin | `vllm_bnb_plugin/bitsandbytes_loader.py` |
| [bnbinvariant.patch](vllm/bnbinvariant.patch) | plugin | `vllm_bnb_plugin/quantization/linear.py` |

## FP32 LoRA

Invar trains and serves LoRA adapters in FP32, so inference reads the adapter that learning wrote. Stock vLLM accepts only FP16 and BF16 adapters. `fp32.patch` adds `float32` to the LoRA dtype setting and lets the shrink and expand operators read FP32 adapter weights. FP32 dot products use IEEE precision, and the adapter delta is rounded to the base output dtype before it is added.

When `VLLM_BATCH_INVARIANT` is set, FP32 shrink runs in a separate kernel, `_lora_shrink_rows`. Each program computes one rank of one token row with FP32 FMA over blocks of 1024 columns and a four warp reduction, so every row takes the same operations whatever the batch size, order or adapter grouping.

`fp32padding.patch` applies on top. CUDA Graph replay leaves stale adapter routes in padding rows beyond the current token count, and the kernel now skips every row past the live count.

## NF4 plugin

`bnb.patch` replaces the loader's call to `get_rename_mapper()`, which vLLM 0.28.0 lacks, with a `dataclasses.replace` copy that keeps the renaming rules and clears the stacked map.

`bnbinvariant.patch` is the NF4 dispatch change of the admitted candidate. The stock plugin calls `matmul_4bit`, which can pick a different kernel when the row count changes. The patch dequantizes each shard with `dequantize_4bit` and multiplies with `torch.nn.functional.linear`, one path for every row count.

## Pinning and applying

Each patch has a `before` and an `after` manifest listing the SHA-256 of every touched file at the pinned revision and after patching. Checking both proves the tree started and ended at the pinned bytes. The `fp32padding` before manifest equals the `lora_shrink_op.py` line of the `fp32` after manifest.

With `invar_root` set to this repository, run each patch in order (`fp32`, `fp32padding` in the tree containing `vllm/`; `bnb`, `bnbinvariant` in the plugin checkout):

```sh
shasum -a 256 -c "$invar_root/patches/vllm/fp32.before.sha256"
patch --batch --forward -p1 -i "$invar_root/patches/vllm/fp32.patch"
shasum -a 256 -c "$invar_root/patches/vllm/fp32.after.sha256"
```

Then install the plugin with pip, which registers it through `vllm.general_plugins`. At run time the worker hashes the loaded LoRA modules ([worker/vllm/profile.py](../worker/vllm/profile.py)) and plugin sources ([worker/vllm/quantization.py](../worker/vllm/quantization.py)) into its reported identity, so every admitted result is bound to the patched bytes. [invar.cabal](../invar.cabal) ships the patches and manifests as source files.
