# Model execution feasibility

The standalone [probe](../worker/probe.py) validates a narrow model execution path. It is not a second runtime, a checked numerical adapter or a reinforcement-learning implementation. Python dependencies are isolated from the Haskell package.

## Reproduction

Use Linux x86-64, an isolated Python 3.12 environment with pip, a CUDA-compatible NVIDIA driver and sufficient available GPU and host memory. The dependency file selects the CPython 3.12 Linux CUDA 12.8 PyTorch wheel explicitly. The measured host used Ubuntu 24.04, Python 3.12.3 and driver 570.211.01. The following paths are examples: replace them with a writable directory that you own under `/data`, and an existing isolated Python environment. Do not reuse another project's environment or output directory.

```sh
INVAR_ROOT="/data/your-directory/invar"
INVAR_PYTHON="$INVAR_ROOT/env/bin/python"
mkdir -p "$INVAR_ROOT/cache" "$INVAR_ROOT/tmp" "$INVAR_ROOT/runs"
export HF_HOME="$INVAR_ROOT/cache/huggingface"
export PIP_CACHE_DIR="$INVAR_ROOT/cache/pip"
export TORCH_HOME="$INVAR_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$INVAR_ROOT/cache/extensions"
export TRITON_CACHE_DIR="$INVAR_ROOT/cache/triton"
export CUDA_CACHE_PATH="$INVAR_ROOT/cache/cuda"
export TMPDIR="$INVAR_ROOT/tmp"
export HF_ENABLE_PARALLEL_LOADING=false
export HF_XET_NUM_CONCURRENT_RANGE_GETS=1
"$INVAR_PYTHON" -m pip install --requirement worker/requirements.txt
"$INVAR_PYTHON" -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", cache_dir=sys.argv[1], max_workers=1, token=False)' "$INVAR_ROOT/models"
INVAR_RUN="$(mktemp -d "$INVAR_ROOT/runs/probe.XXXXXX")"
systemd-run --user --scope -p MemoryMax=32G -p MemorySwapMax=0 -p CPUQuota=400% env HF_HUB_OFFLINE=1 "$INVAR_PYTHON" worker/probe.py --cache "$INVAR_ROOT/models" --output "$INVAR_RUN/state" --prompt "What is 19 multiplied by 23? Explain briefly." --tokens 32 --seed 17
```

The measured model process had a 32 GiB host-memory limit, no swap and four CPU cores. These limits do not reserve GPU memory; check available resources before running. The probe requires the complete pinned model snapshot, uses local-only loading and refuses an existing output directory. It does not download on a cache miss or retry with a different model or precision. Keep the generated reports and checkpoint files from each attempt.

## Numerical configuration

Model: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The same base and unmerged adapter are used for sampling and learning. The tested stack was PyTorch 2.11.0+cu128, Transformers 5.16.1, PEFT 0.20.0, bitsandbytes 0.50.2, Accelerate 1.14.0, huggingface-hub 1.30.0, safetensors 0.8.0 and NumPy 2.5.2.

| Component | Configuration and observed scope |
|---|---|
| Frozen base | NF4 with double quantization, uint8 packed storage and requested BF16 linear compute; non-quantized parameters remain BF16. The parameter report contained 606 frozen uint8 and 578 frozen BF16 tensors. |
| LoRA | Rank 8, alpha 16, dropout 0, no bias adaptation; gate/up/down FFN projections in 64 language-model layers. All 384 trainable tensors across 192 targets were FP32. |
| Forward and backward | Eager attention, Hub kernels disabled, full-context recomputation with `use_cache=False`, non-reentrant gradient checkpointing. No wholesale FP32 promotion of the frozen base. |
| Sampling | Temperature 0.8; logits are converted to FP32 before softmax. A request-seeded CUDA generator samples from those weights; each selected token retains its normalized-weight log probability. |
| Objective | Response-token cross entropy, with prompt labels ignored. This is a feasibility loss, not GRPO or a reward-driven update. |
| Gradients and optimizer | Gradients on the FP32 adapter must all exist and be finite; individual optimizer moment dtypes are not separately logged. The diagnostic squared-gradient sum uses FP64. AdamW uses learning rate 1e-4, betas 0.9/0.999, epsilon 1e-8, zero weight decay, `foreach=False` and `fused=False`. |
| Backend controls | Float32 matmul precision is set to `highest`; cuDNN TF32 is disabled. Individual fused-kernel accumulation and reduction schedules were not independently verified. |
| Quantization state | The frozen base is loaded and quantized once in this process. The continuation comparison reuses that base; it does not certify a newly quantized base after a process restart. |

Configured dtypes and parameter reports are not a proof of every intermediate or kernel accumulation format. The probe rejects non-finite loss, missing or non-finite trainable gradients, zero gradient norm, non-finite updated parameters and an unchanged first-step adapter. It checks the loading diagnostics exposed by the selected library API, not every internal conversion diagnostic.

## Measured result

One successful run on an NVIDIA Ada workstation GPU with 48 GiB of device memory generated 32 response tokens from a 26-token prompt using seed 17. The answer reached the token limit before completing its explanation; it is not a successful-answer or quality result. The process exited successfully after approximately 190.2 seconds. Measurements were taken on shared hardware, not an isolated benchmark, and there were no repeated performance trials.

| Stage | Seconds | Peak allocated bytes | Peak reserved bytes |
|---|---:|---:|---:|
| Load | 160.5120 | 18057402368 | 18266193920 |
| Generate 32 tokens | 10.5129 | 18139983360 | 18341691392 |
| First update | 2.1897 | 18492341760 | 19077791744 |
| Uninterrupted second update | 1.3913 | 18731395584 | 19363004416 |
| Restore | 0.5114 | 18766305792 | 19363004416 |
| Resumed second update | 1.6319 | 18731395584 | 19363004416 |

These are synchronized PyTorch stage measurements. An update includes the loss forward pass, backward pass, optimizer step and validation; backward-only latency was not measured separately. Device allocation statistics do not include every driver or context allocation. Saving, state comparison and other host work contribute to the full process time but are not all represented by table rows.

The first update produced loss 0.2909276485 and gradient norm 0.7575614537; the second produced loss 0.2468045801 and gradient norm 0.7738529867. All 384 optimizer step counters advanced from 1 to 2. Saving and restoring the first-step adapter, optimizer and CPU/CUDA RNG state, then repeating the second update on the same batch, reproduced the uninterrupted adapter and optimizer tensors with equal shape, dtype and raw bytes. Non-tensor state was compared structurally by type and value, not by an asserted universal byte encoding.

Canonical adapter digests were `08a9cd1e26784346709e1bb9df239b18c84424006fd9d4b107a2a97ace62abc4` initially, `de174cbfd37d0cfdacb777c95ee5e4d393dc378fa4fc8420b68a1a5076bfde12` after the first update, and `d53e8e49446dc3743cca681b12895f438ebda9df4d6253b504423d87bce39c3a` after both versions of the second update. These identify sorted tensor content and metadata, not container-file bytes. The probe's `continuation.published` field names the saved first-step adapter; it is not a durable publication receipt or authorization.

Two earlier attempts failed explicitly: an incomplete local snapshot was rejected before model loading, and a loading-report serialization error stopped another attempt after weight loading. Completing the same pinned snapshot and converting the library's diagnostic sets to sorted lists resolved those failures without changing the selected numerical profile. The reported successful version also compares tensor bytes, including signed zero, rather than treating ordinary tensor equality as bitwise equality.

## Evidence limits

The unchanged probe source corresponding to this result has SHA-256 `d24aeb55ca2ec3e2705bd7a27f58d0d11231243007b2e4adc35d8d61e0450855`. This is one same-process continuation comparison, not history noninterference, cross-hardware equivalence, process-restart recovery, kernel certification or training effectiveness. Model artifacts and generated checkpoints are not included in the source change.

Remaining integration work includes binding the worker to checked semantic inputs, reward-driven learning with distinct probability roles, durable publication and actual subsequent consumption, paired physical histories, and performance and held-out quality baselines. The result establishes a feasible generation-and-update path; it does not complete the inference-and-RL release.
