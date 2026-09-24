# vLLM backend

The vLLM backend runs rollout inference on CUDA. It drives a patched vLLM engine inside one Python process, reports what it loaded and consumed over the checked worker protocol, and waits for the Haskell core to grant permission before any model or sampler call. The release acceptance ran Qwen/Qwen3.8-27B on one GH200 with an NF4 base and FP32 LoRA on every MLP projection. Learning runs in a separate PEFT process that reads the same policy tensors. The protocol is described in [the worker contract](../design/contract.md) and the training loop in [the loop design](../design/loop.md). The Apple Silicon backend is described in [the MLX backend](mlx.md).

## Entry points

Each file under `entries/` imports one module from `worker/vllm/`.

| Entry | Protocol |
| --- | --- |
| `entries/vllminspect.py` | Loads the engine and adapter, prints one `identified` record with the four identities, exits |
| `entries/vllminfer.py` | One checked request for `invar infer` |
| `entries/vllmsession.py` | Serial requests on one engine, one permission each |
| `entries/vllmbatch.py` | One finite group of requests under one atomic permission |
| `entries/vllmresident.py` | Finite groups across cohorts and policy updates on one engine |
| `entries/vllmscore.py` | Scores a prescribed token path, with optional probes of the full vocabulary |

Every entry takes `--config`, a launch configuration, and `--cache`, the Hugging Face cache. The core passes the configuration path through `--worker-config` for `invar infer`, `invar evaluate` and `invar score`, and through `--inference-config` for `invar train`. Diagnostics go to stderr; stdout carries only protocol JSON.

## Building the engine

The engine is vLLM 0.28.0 with four pinned patches, applied in a dedicated environment and checked against before and after SHA-256 manifests. [The patch guide](../../patches/README.md) lists them.

1. `fp32.patch` lets the Triton LoRA shrink and expand kernels keep FP32 adapter weights. With batch invariance enabled, each shrink row uses a fixed sequence of FP32 fused multiply adds over 1024 columns and a fixed reduction tree, so batch size, request order and adapter grouping never change the reduction.
2. `fp32padding.patch` makes the row kernel skip padded rows that CUDA Graph metadata can leave behind.
3. `bnb.patch` adapts the official `vllm-bnb-plugin` to the vLLM 0.28.0 weight mapper.
4. `bnbinvariant.patch` makes NF4 matrix multiplication take one path for every row count: dequantize each packed shard, then call `torch.nn.functional.linear`.

The process runs with `VLLM_BATCH_INVARIANT=1`, `VLLM_LORA_ENABLE_DUAL_STREAM=0` and `VLLM_ENABLE_V1_MULTIPROCESSING=0`, the last keeping the engine core inside the worker process. Before constructing the engine, `worker.vllm.entry.factory` calls `worker.vllm.gdn.install()`. This replaces the recurrent update of the Qwen gated delta network layers with the Triton kernel in `worker/vllm/recurrence.py`, which runs the same FP32 arithmetic for prefill and decode, so a sequence reaches the same state however the scheduler splits its tokens. The layer requires CUDA, batch invariance and FP32 recurrent state, with speculative decoding off.

`worker/vllm/configuration.py` reads the launch file: the model, a pinned `revision`, the SHA-256 of the PEFT receipt as `handoff`, and `engine`, the arguments for vLLM's `LLM`. Checkpoint mode adds `template`, the PEFT directory that a checkpoint adapter must match. The release acceptance used BF16 activations, `bitsandbytes` quantization and loading, no prefix caching, one device, FP32 LoRA on `gate_up_proj` and `down_proj`, `processed_logprobs` and FP32 recurrent state. `entries/handoff.py` exports a learner adapter as an unmerged FP32 PEFT directory and writes that receipt.

## FP32 LoRA materialization

`worker/hf/artifact.py` reads a PEFT handoff against the caller's receipt digest, or reads a checkpoint `adapter.safetensors` against the template's complete tensor schema. `worker/vllm/lora.py` turns the package into resident vLLM buffers. `qwen_mlp_targets` builds the target map from the engine's layer count: `gate_proj` and `up_proj` pack into `gate_up_proj`, and `down_proj` maps to itself, 192 source modules for the 64 layer model. Rank, alpha and rsLoRA scaling come from the PEFT configuration; B is multiplied by the scaling in FP32 and the unused rank is zero padded.

`activate` registers and activates the adapter, then compares every FP32 word of the resident slot, including padding, with values derived independently from the source tensors. `worker/vllm/worker.py` runs these checks inside the engine's worker through `collective_rpc` and confirms that the runner and LoRA manager own the same model and that the sampler reports `processed_logprobs`.

## Identities

Every request names four identities, and the worker reports the values it observed.

| Identity | Computed by | Covers |
| --- | --- | --- |
| `adapter` | `worker/vllm/lora.py` | The FP32 policy tensors |
| `tokenizer` | `worker/hf/operation.py` | The effective text operation: template, special tokens, EOS, prompt and decode settings |
| `base` | `worker/vllm/identity.py` | Every parameter, `state_dict` entry and buffer, plus NF4 quantization states and shard offsets |
| `assembly` | `worker/vllm/profile.py` | The engine description together with the policy's configuration, target map and transformation |

The assembly description records the vLLM configuration, library versions, device, engine flags, every module's class and forward methods, the sampler, the LoRA buffer shapes, and hashes of the BitsAndBytes plugin sources and CUDA library. It also hashes the files of `SOURCE_MODULES` in `worker/vllm/profile.py`, the Invar checking modules and the vLLM modules that choose arithmetic or scheduling, and of `ENTRY_FILES`, the entry and protocol files. Changing any of these bytes changes the assembly digest.

The learner has its own assembly, format `invar-model-assembly-v3`, whose `implementation` field hashes every file in `SOURCE_FILES` of `worker/implementation.py`. The core receives both: `--base-digest` and `--assembly-digest` name the learner, `--behavior-base-digest` and `--behavior-assembly-digest` name the vLLM rollout.

The base identity reads all current bytes of the model state, with sixteen threads hashing independent tensor streams. It is read once when a policy is prepared or activated; that reading feeds the load, consumption and result reports of the next grant, and the monitor reads base and assembly once more at completion. A result is delivered only if that last reading matches.

## Checks at every step

`worker/vllm/execution.py` enqueues the requests without running them and binds each one to its prompt, EOS and LoRA selection. The `begin` RPC installs a `Monitor` from `worker/vllm/state.py` with no permission. The core's permission arrives, the `permit` RPC rechecks the resident policies, and only then may a model step run.

The monitor wraps the runner's `_model_forward` helper, so eager calls and CUDA Graph replays are both observed. Before each model call it checks the scheduled requests and token counts, the LoRA slot of every token and every routing tensor the LoRA kernels read (`worker/vllm/mapping.py`), and the token IDs and positions against the prompt plus the tokens already sampled. It then checks that the logits processor reads the last hidden row of each request, that sampling is random with no filters or penalties at the requested temperatures, and that each request uses its own generator seeded with its own seed. After the sampler it records each selected token and its FP32 log probability. Each returned trajectory must end at EOS or at its limit and agree token for token and word for word with the monitor's record.

## Batch execution and the admitted candidate

`invar evaluate --worker-mode batch`, `invar train --inference-mode batch` and `invar infer batch` send a group of calls with format `invar-inference-batch-v1`. Each call keeps its own program, load, binding and request. `execute_batch` in `worker/vllm/runtime.py` prepares every member, emits one `consumed` frame with all members' reports, receives all permissions in one line, runs one native generation and emits one `result` frame. An invalid member means no member runs.

Stock vLLM can give the same prompt slightly different numbers depending on which requests share its batch, because the order of floating point reductions follows the batch layout. The reference backend for the release acceptance was vLLM with the FP32 LoRA patches. The candidate adds two changes. NF4 matrix multiplication always takes the same kernel path, through the [NF4 dispatch patch](../../patches/README.md#nf4-plugin). Softmax is computed by a kernel that reduces each row on its own in fixed blocks of 1024 columns, so each row's result is independent of the other rows in its batch. [The acceptance results](../results/acceptance.md) compare the two on sealed units, and [the performance results](../results/performance.md) give the overhead against the stock engine.

## Resident inference

`entries/vllmresident.py`, selected with `--worker-mode resident` or `--inference-mode resident`, keeps one engine for the whole workload. Each group runs as a finite batch. After the core admits the results, `release` in `worker/vllm/residency.py` checks that no request, cache block or sampler row remains before acknowledging. When the next group names the same policy the worker verifies it in place; when it names a new one the worker checks the new package, confirms base and assembly are unchanged, removes the old registration and activates the new one under a fresh adapter ID.

## Native learning composition

`invar train` runs rollout and learning as two processes: `--inference-python` and `--inference` select the vLLM side, `--python` and `--learning` the PEFT learner, which loads the same model as NF4 with FP32 LoRA. Both share the policy and tokenizer identities and report their own base and assembly.

The vLLM process reads each cohort's policy from the loop's current `adapter.safetensors` in checkpoint mode. The sampled behavior probabilities travel to the learner as exact FP32 words and stay separate from the proximal, current and reference probabilities the learner computes; the request's `behavior_model` field records which base and assembly produced them. Each update publishes a checkpoint, the next cohort consumes that published adapter, and after the last cycle an independent `entries/vllminfer.py` process loads the final checkpoint.

## Scoring and probes

`invar score --worker entries/vllmscore.py` runs a target policy along the token path of an admitted source inference. The monitor runs with prescribed paths from `worker/vllm/prescribed.py`: the native sampler still computes its processed probabilities, and the scorer replaces the selected token with the prescribed one before vLLM records it. With `--probe-steps`, `worker/vllm/distribution.py` observes the sampler's softmax over the vocabulary and copies the complete FP32 probability vector at the selected steps.
