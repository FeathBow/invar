# Performance

These results were measured on earlier code and have not been measured again since. On CUDA the core and workers were those of commit `b64985c`, run from a tree recorded as `6b535ab` before the history was rewritten; the code is identical. On Apple Silicon the core and the inference workers were those of commit `974dbe8`, and the training cycles ran the workers of commit `ac1f339` (numerics profile v5), which produced the training inputs. Later code changed where the reference policy is scored, from the learner to the rollout, and computes each role's identity when a model loads; neither has been measured. On the measured code, the training loop orchestrated by the Invar core cost at most 1.1 times two baselines: the direct route, which runs the same numerical workers from a Python runner that calls the core only for helper steps, and the native route, which uses the backend library directly. Budgets fixed in advance decided it per platform and scope, on one GH200 and on Apple Silicon. See [the loop design](../design/loop.md) and [the release acceptance](acceptance.md).

## Routes and scopes

| Route | What runs |
| --- | --- |
| invar | the core drives the workers: it plans every call, checks inputs and outputs, computes rewards and update inputs, and publishes each successor policy |
| direct | the same numerical workers, with the same physical owners, load counts and cohort barriers, driven by a Python runner that calls core helper commands to plan calls, build update inputs and publish policies; those calls count toward its time and memory |
| native | the backend library used directly: `vllm.LLM` generation with a resident PEFT learner on CUDA, `mlx-lm` batch generation with native AdamW on MLX; the same core helper commands plan calls, score responses, build update inputs and publish policies, and in cycles its learner computes the reference probabilities |

Invar over direct measures the added cost of the complete Invar orchestration over that runner; the native route chooses its own numerical settings and batching. The workers are described on the [vLLM](../backends/vllm.md) and [MLX](../backends/mlx.md) backend pages.

| Scope | Workload |
| --- | --- |
| inference | evaluation of a fixed policy over a frozen evaluation workload |
| cycles | complete training cycles: generation, reward and update input construction, the learner update, publication of the successor and its activation; two cycles of 128 rollouts on CUDA and of 64 rollouts on MLX |

## Metrics

Each route is observed from launch of its entry process until its main child is reaped.

| Metric | Meaning |
| --- | --- |
| complete process | seconds from launch to reap |
| response tokens | response tokens counted from the retained generation records |
| tokens per online second | response tokens divided by complete process seconds |
| memory | largest sampled value: on CUDA the GPU memory used and the RSS of the owned process tree; on MLX the physical footprint of the owned process tree |

MLX uses the physical footprint that `top` reports because unified memory makes RSS count pages the GPU holds once the CPU touches them: the worker's identity hash alone raises RSS by 6 GiB while the footprint stays the same.

## Budgets and decision rule

Budgets are fixed in a file before the blocks they govern run; both platforms use the same ratios of invar over each baseline.

| Metric | Budget |
| --- | --- |
| complete process | at most 11/10 |
| memory | at most 11/10 |
| response tokens | at least 10/11 |
| tokens per online second | at least 10/11 |

Each block runs two repeats: repeat 0 in the order invar, direct, native, and repeat 1 in the order native, direct, invar. Each comparison takes the median over the two repeats of the ratio in each repeat. A block is MET only if every comparison meets its budget, and NOT_MET if any misses. A missing record, a failed process, a zero denominator or an absent budget makes the block UNKNOWN. Each decision was recomputed from the retained records with identical results.

## CUDA results

Both CUDA blocks are MET on all ten comparisons: five metrics against each baseline.

| Block | Decision | invar / direct | invar / native |
| --- | --- | --- | --- |
| inference | MET 10/10 | process 0.964, tok/s 1.038, RSS 0.933 | process 0.781, tok/s 1.278, RSS 0.854 |
| cycles | MET 10/10 | process 1.066, tok/s 0.938, RSS 0.974 | process 1.020, tok/s 0.990, RSS 0.945 |

The other ratios are response tokens 1.000 and GPU memory 1.002 against direct and 0.998 and 0.999 against native for inference, and 1.000 and 1.000 against direct and 1.008 and 1.018 against native for cycles. Invar and direct generate the same response tokens in every repeat (23385 for inference, 24147 for cycles). The 0.781 against native reflects the native evaluator's longer load and close phases. In cycles invar takes 820 to 822 s per repeat and direct 765 to 774 s.

## MLX results

Both MLX blocks are MET on all eight comparisons: four metrics against each baseline.

| Block | Comparison | complete process | response tokens | tokens per online second | footprint max |
| --- | --- | ---: | ---: | ---: | ---: |
| inference | invar / direct | 0.995 | 1.000 | 1.005 | 1.000 |
| inference | invar / native | 1.073 | 0.997 | 0.929 | 1.002 |
| cycles | invar / direct | 0.989 | 1.000 | 1.012 | 1.004 |
| cycles | invar / native | 1.017 | 1.000 | 0.983 | 1.006 |

For inference, invar takes about 1016 s per repeat, direct 1015 and 1027 s, native 952 and 941 s. Invar and direct generate 25629 response tokens in both repeats and native 25709.

For cycles, invar takes 7182 and 7150 s per repeat, direct 7142 and 7354 s, native 7034 and 7057 s. All three routes generate 25842 response tokens in both repeats. The footprint peaks at about 18.1 GiB for invar and 18.0 GiB for both baselines.
