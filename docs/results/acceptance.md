# Release acceptance

The release acceptance admits a new inference implementation of `Qwen/Qwen3.8-27B` as a replacement for the reference in training loop rollouts and evaluation. The core decided it from one use contract on retained records from one GH200; this page walks through that contract. See also [the contract design](../design/contract.md) and [the loop design](../design/loop.md).

## Reference and candidate

The reference is vLLM with the FP32 LoRA patches, as used in the CUDA training runs. The candidate adds two changes so that the numbers of each request do not depend on which requests share its batch: NF4 matrix multiplication always takes the same kernel path, and softmax reduces each row on its own in fixed blocks of 1024 columns. Both run the same model revision, tokenizer and frozen base weights at the initial adapter `08a9cd1e…`. The [vLLM backend page](../backends/vllm.md) describes the CUDA worker.

## Population and sealed draws

The declared population is the GSM8K training split after exclusions: 7473 word problems whose prompts ask for a final line `#### <number>`, minus the 16 prompts used during development and every key an earlier draw exposed. A unit is one prompt with seed 17, sampled at temperature 0.8 with at most 256 response tokens in a context of 1024.

Each stage draws its units through a commitment written before any sealed item is read, binding the pool, exclusions, contract and exposure ledger by digest. The draw orders the eligible keys by the SHA-256 digest of the committed seed and the key, takes the first units, and refuses a reused commitment or a changed ledger. Every draw appends its keys to the ledger, and later commitments exclude them. The main draw took 2000 keys with seed 3 from the 7329 still eligible, after excluding the 16 development prompts and the 128 keys of two probe draws.

## Two stages

| Stage | Units | Executions per unit | Role |
| --- | ---: | --- | --- |
| probe | 64 | reference, candidate, one candidate repeat under another partition | measures the values the main contract needs |
| main | 2000 | reference, candidate, two candidate repeats under other partitions | decides the admission |

The main stage runs 32 groups of 64 requests per side; the two candidate repeats shift the groups by 16 and 32 positions.

## Measurement

The measurement is the `exact-decimal/v1` method, a checked scalar program the core runs on each response. It compares the final line with the expected answer as exact rational numbers; the loss is one minus that reward, and a response that reaches 256 tokens counts as loss one. The core averages loss within each unit, then across units, in exact arithmetic.

## Requirements

| Requirement | Value |
| --- | --- |
| reference loss, Bernstein upper bound | at most 61/200 |
| loss increase over the reference, Bernstein upper bound | at most 1/40 |
| confidence | alpha 1/40 for each bound, within a family alpha of 1/20 |
| invariance | `behavior-bits` equal across three candidate executions under different batch partitions |

The bounds use the empirical Bernstein rule of Maurer and Pontil (2009, Theorem 4): for n units with unbiased variance s², the width is `sqrt(2 s² ln(2/alpha) / n) + 7 r ln(2/alpha) / (3 (n - 1))`, with r equal to 1 for reference loss and 2 for the paired increase. The core encloses the logarithm and square root upward with integer arithmetic. The `behavior-bits` relation requires equal tokens and bitwise equal FP32 words of the reported behavior probabilities between successive candidate executions.

## How each value was derived

Every value follows from the probe or from a stated convention.

| Value | Source |
| --- | --- |
| units n = 2000 | the probe took 1.98 s per unit execution, so seven hours afford 4239 units; 2000 units at three executions fit under four hours |
| regression ceiling 1/40 | the Bernstein width at n = 2000 with the variance taken as the rule of three bound 3/64 on the probe's zero correctness discordance: 0.0246, rounded up |
| reference ceiling 61/200 | the probe's reference loss 0.2656 plus the upper Bernstein width at n = 2000 |
| family alpha 1/20, split evenly | the conventional 95 % family level |

The main run's observed paired variance, 0.0205, stayed below the 3/64 the probe supplied, which is why the loss increase bound fits inside its ceiling.

## Results

The contract bound the reference as assembly `b13d1c95…` and the candidate as assembly `7fa845df…`. The main stage decision is **admitted under declared reliance**.

| Stage | Units | Result |
| --- | ---: | --- |
| probe | 64 | correctness discordance 0/64, token discordance 21/64, bitwise repeat 64/64; grounds every main stage value |
| main | 2000 | admitted under declared reliance |

| Quantity on 2000 units | Value | Requirement |
| --- | ---: | ---: |
| reference loss | 0.2635 | |
| candidate loss | 0.2640 | |
| loss increase (21 units worse, 20 better) | 0.0005 | |
| reference loss, Bernstein upper bound, alpha 1/40 | 0.2978 | at most 61/200 |
| loss increase, Bernstein upper bound, alpha 1/40 | 0.0202 | at most 1/40 |
| candidate behavior bits equal across three executions under different partitions | 2000/2000 | exact |
| units whose tokens differ from the reference | 649/2000 | diagnostic |

On both sides 497 responses reached the token limit and count as loss one.

The contract raises 14 kinds of external premise, such as isolation of the sealed data from candidate search, amounting to 42,017 concrete conditions over the 2000 units. It declares one reliance entry per kind, each with a named authority and the SHA-256 digest of its basis files, and the decision retains every condition with the entry that covers it.

## Rerunning an admission

A reader reruns the admission from retained records without executing a model.

```sh
invar use inspect --contract contract.json --runs runs.json
invar use admit --contract contract.json --runs runs.json
```

The contract is the complete `invar-use-contract` document, including domain and reliance entries. The runs file is a JSON array with one entry per unit, `[cohort index, input key, paired arguments, repeated arguments]`. The paired arguments are the input options of `invar compare numerical` without `--relation` or `--budget`, prefixed once with `--reference-` and once with `--candidate-` (digests, request, binding, log and exit code). Each repeated array uses the same options unprefixed for one further candidate execution:

```json
[0, "sealed/0/seed/17",
  ["--reference-assembly-digest", "b13d1c95…", "…", "--reference-log", "reference-0/stdout", "--reference-exit-code", "0",
   "--candidate-assembly-digest", "7fa845df…", "…", "--candidate-log", "candidate-0/stdout", "--candidate-exit-code", "0"],
  [["--assembly-digest", "7fa845df…", "…", "--log", "repeat-0/stdout", "--exit-code", "0"], ["…"]]]
```

`inspect` prints the observation; `admit` adds the finding, the decision and the confidence bounds. Exit zero means a judgment was computed, and `decision.status` holds admission under declared reliance, violation or unknown. The decision and bounds above are the output of `invar use admit` on the retained main stage records.

The loop's cost is reported in [the performance results](performance.md).
