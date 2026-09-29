# MLX backend

The MLX backend runs Invar's inference and learning on Apple Silicon. One Python process loads the model on the Metal device, serves rollout groups and learner updates in turn, and reports each operation over the same checked protocol as the CUDA backend: a load report, a consumption report, a wait for the core's permission, then a result bound to its call, attempt and instance. The protocol is described in [the worker contract](../design/contract.md) and the training loop in [the loop design](../design/loop.md). The CUDA backend is described in [the vLLM backend](vllm.md).

## Model and environment

[The model loader](../../worker/mlx/model.py) pins `mlx-community/Qwen3.8-27B-4bit` through its `MODEL` and `REVISION` constants and loads it from the local Hugging Face cache on the Metal device, with all 64 layers. The weights are affine 4 bit with group size 64. `worker/mlx/adapter.py` adds FP32 LoRA with rank 8 and scale 2.0 to `gate_proj`, `up_proj` and `down_proj` of every layer, 192 projections, and freezes everything else. [The lock file](../../worker/locks/mlx.txt) pins CPython 3.14, MLX 0.32.1 and MLX-LM 0.31.3 with hashes.

An optional runtime configuration with format `invar-mlx-runtime-v1` sets `batch_size`, the decode batch of MLX-LM's `BatchGenerator`, `prefill_step`, its prefill chunk, `cache_bytes`, the free buffer cache of the allocator, and `numerics`, the profile every entry loads: `primary`, the row independent profile, or `native`, the library projection, LoRA and attention with the project's cache and recurrence corrections. The defaults are 1, 512, 256 MiB and `primary`. Each entry reports the profile it loaded in its `profile` record.

## Entry points

| Entry | Use |
| --- | --- |
| `entries/mlxoperation.py` | Prints the tokenizer operation digest |
| `entries/mlxinitialize.py` | Writes the initial `adapter.safetensors` and `learner.pt` |
| `entries/mlxinfer.py` | One checked request for `invar infer` |
| `entries/mlxbatch.py` | One finite group for `invar infer batch` and batch evaluation |
| `entries/mlxcohort.py` | Resident inference across cohorts for `invar evaluate --worker-mode resident` |
| `entries/mlxresident.py` | Shared owner of inference and learning for `invar train` |
| `entries/mlxscore.py` | Scores a prescribed path |
| `entries/mlxcodec.py` | Reads native learner containers for `invar inspect` |

The initializer reports identities for two roles. Its final `initial` record gives the adapter identity as `policy`, the `learner` file identity, the `tokenizer`, and the learning role's `base` and `assembly`, which `invar train` declares for the learner. Its `profile` record gives the inference role's `base` and `assembly` under `inference`; `invar infer`, `invar policy` and `policy.json` use those, and the learning `assembly` is refused there.

## Why rows must be independent

A cohort's behavior probabilities must not depend on how its requests were grouped. The same prompt has to produce the same token and the same FP32 probability word whether it runs alone or with other requests, in any order. On Apple Silicon the library's quantized matrix product and attention choose different kernels as the number of rows or queries in a call changes, and different kernels round differently. The changed last bits then change sampled tokens further along the response.

The row independent profile fixes the shape every kernel sees in calls whose size depends on the batch, and keeps the library kernels for calls whose size depends only on one request.

## Numerics profiles

`worker/mlx/numerics.py` defines two profiles. Each names the classes swapped in for the library's `nn.QuantizedLinear`, `LoRALinear` and `Qwen3NextAttention`, in place and without copying weights.

| Profile | Projection | LoRA | Attention | Learning projection |
| --- | --- | --- | --- | --- |
| `native-library-arithmetic/v4` (stock) | `nn.QuantizedLinear` | `LoRALinear` | `Qwen3NextAttention` | same |
| `independent-native-rows/v8` (primary) | `RowLinear` | `ColumnLoRALinear` | `QueryAttention` | `nn.QuantizedLinear` |

The stock profile runs the library kernels and serves as the ordinary native baseline. The primary profile has three parts.

`RowLinear` in `worker/mlx/projection.py` calls the library projection for 64 rows or more. Below 64 rows it pads with zeros to a multiple of 4, runs `mx.quantized_matmul` on a stack of 4 row blocks and discards the padded outputs. Every row below 64 meets the same kernel, so its output is the same for any row count, order or split.

`ColumnLoRALinear` computes the adapter delta as matrix products over input columns. A single column is repeated to 2 and the extra output discarded, so single rows also use matrix arithmetic.

`QueryAttention` in `worker/mlx/attention.py` uses the library attention for one sequence of 64 or more queries. Otherwise it rotates left padded keys to canonical positions and evaluates queries in blocks of 8 with `mx.fast.scaled_dot_product_attention`, repeating the last query to fill a short final block and discarding the repeats.

Generation keeps every call whose size depends on the batch below 64. With `RowLinear` installed, `prefill_batch_size` in `worker/mlx/rollout.py` prefills one request per call, so a prefill chunk that reaches the library kernels has a shape fixed by that request's prompt and the prefill step. Decode sends one row per request, and a configured `batch_size` of 64 or more is rejected, so decode always runs in the 4 row and 8 query blocks.

Both profiles also install the cache and recurrence corrections. `worker/mlx/cache.py` keeps key and value dtypes when an empty cache joins an occupied one and builds recurrent masks from both valid lengths and left padding. `worker/mlx/recurrence.py` runs the gated delta update in checkpointed segments of 16 tokens.

## Assembly identity

`identities(loaded, role)` in `worker/mlx/model.py` computes four identities from the loaded model for the inference or the learning role: the digest of the FP32 LoRA tensors, the tokenizer operation digest, the base (format `invar-mlx-base/v1`, the model configuration and every parameter outside the adapter), and the assembly.

The assembly is the SHA-256 of a description with format `invar-mlx-assembly/v3`: the layer count, the model class, each LoRA target's class, scale, shapes and quantization, and the numerics description from `Profile.observe(model, role)`. `observe` first checks the actual model: every projection, LoRA, attention, recurrence and cache module must have the class its profile selects, and every projection must be affine 4 bit with group 64. It then records the profile name, the module inventory, the block sizes and padding rules, the recurrence segment length, the `mlx` and `mlx-lm` versions, and the hashes of the files the role depends on, listed with their reasons in `worker/mlx/implementation.py`. The primary profile adds `projection_stock_rows: 64`, `attention_stock_queries: 64` and `prefill: "one request per prefill call"`; the stock profile carries none of these fields. The learning role adds the learning projection and where each probability role comes from.

Because the description comes from the installed modules, the two profiles share a base digest and have different assembly digests, and the inference and learning roles share a base digest and have different assembly digests. Initialization reports the inference identity in its `profile` record, since the checkpoint carries the learning identity, and the shared resident checks each inference group against the learner's saved successor with the inference assembly. The worker computes the identities when it activates a policy and again before each group reports consumption, and the core grants permission only when they match the plan.

## Inference

`worker/mlx/rollout.py` drives `BatchGenerator` with one `Sampler` per request. The sampler turns the FP32 logits into a distribution with a precise softmax at the requested temperature, draws a token with a key derived from the request's seed, and records the log of the selected token's probability. When the generator returns the token, the sampler checks that it is the one it drew. Each trajectory stops at the tokenizer's EOS or at its limit. Requests use their own keys, and `worker/mlx/inference.py` fails a group if the global MLX random state that the learner saves moved during it.

## Learning

`worker/mlx/step.py` and `worker/mlx/learning.py` follow the step exchange in the [loop design](../design/loop.md). For each update the worker emits `loaded_learner`, emits `consumed` and waits for permission. Before the first optimizer step it reports the proximal words of the samples outside that step. In each step it takes the samples in the declared order, evaluates the learner graph at the rollout temperature, reports the current words and applies the objective and reward cotangents the core returns, accumulating both gradients. Each step ends with one AdamW step and an `applied` report. The probabilities, the first step's gradients and the successor checkpoint are written after the last step.

With the primary profile, `learning` in `worker/mlx/training.py` switches `RowLinear` back to `nn.QuantizedLinear` for the learning computations and restores it afterwards, including on failure. Each learning call carries one complete trajectory, so its row count is fixed by that trajectory. `worker/mlx/backward.py` retains every decoder layer's input and evaluates one layer's VJP at a time. The VJP's own forward values are the proximal and current probabilities the learner reports.

## Checkpoints and RNG state

`adapter.safetensors` holds the `.lora_a` and `.lora_b` tensors with metadata `invar_policy: mlx-f32/v1`. `learner.pt` fills the learner slot as an MLX safetensors container: its metadata holds format `invar-mlx-learner/v1`, the four identities, the parameter names and the AdamW configuration, and its tensors hold the optimizer state and the random key. `worker/mlx/state.py` checks the step, learning rate, moments and a single `uint32` key of shape `(2,)`.

MLX exposes its global random state as a read only view. `restore_random` rebuilds the seed from the key's two words, checks that `mx.random.key(seed)` reproduces the saved key and seeds MLX with it. Restoration then compares the optimizer and random state with the file. The core reads these containers through `entries/mlxcodec.py` with `--rng-profile mlx`.

## Resident sessions

Training keeps one model for both roles: `entries/mlxresident.py` for inference and learning with `--inference-mode shared --learning-mode shared`, one executable, one cache and one session. The owner in `worker/mlx/resident.py` keeps the model, current LoRA, fixed reference, optimizer and random state until the core sends close. After an update, the next inference group must name the successor the learner just wrote; the worker verifies the live learner against it without reinstalling. Each later update checks that the live model, optimizer, random state and reference match the saved successor, and fails on any difference.

Each group or update keeps its own load, consumption, permission and result, and ends with a release that the worker acknowledges after clearing the MLX cache and verifying the learner. The core counts one physical load and one close. Fixed policy evaluation uses `entries/mlxcohort.py`, which keeps the model across cohorts and reinstalls the adapter only when a cohort names a different one.

## Measurements and memory

`worker/mlx/metrics.py` synchronizes Metal around each stage and emits `seconds`, `peak_active` and `cache_end` from the MLX allocator. The memory budget uses the process's physical footprint as reported by `top`. Apple Silicon has unified memory, and hashing the weights for the base identity reads GPU pages through the CPU, which makes resident set size count those pages. [The performance results](../results/performance.md) report the footprint and the overhead of the primary profile, and [the acceptance results](../results/acceptance.md) give the verdict of the first acceptance.
