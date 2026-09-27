# The owning training loop

The Haskell core owns the training loop. Each cycle it chooses the policy, declares the cohort, dispatches rollouts, scores the responses, derives the advantages, binds one update, checks the update's artifacts and publishes the next checkpoint. Python workers do the numerical work in between: they load models, generate, compute probabilities and gradients, and step the optimizer. Every worker report passes through the invocation binding described in [contract.md](contract.md), so a run's history is a sequence of checked facts that can be admitted and compared later.

`Invar.Loop` holds the driver. One cycle runs this sequence:

1. Read the current checkpoint's `policy.json` and bind every task to that policy description.
2. Run the cohort through the rollout driver and record each result with its reward.
3. Compile the learning program over the admitted batch, which fixes rewards, groups and advantages.
4. Bind one update to a fresh call, attempt and instance and run it on the learner.
5. Verify the gradients, probabilities, adapter and learner state the worker wrote.
6. Publish the checkpoint under a new name and make it current for the next cycle.

## Cohorts and schedules

`invar train` reads a JSON array of cycles from standard input. Each cycle has `tasks`, `order` and `delivery`, and each task has a name, a group, a prompt, a token limit, a temperature, a seed and an answer. `Invar.Cohort` checks the declaration before anything runs: names are distinct, every group has at least two members so normalization is defined, and every task requests the cohort's policy.

`order` and `delivery` are permutations of member positions. `Invar.Schedule` dispatches members in `order` and hands their results to admission in `delivery`. Admission puts every result back at its member's position, so the learning program always sees the batch in logical order, and a changed schedule changes only when members run and when their results arrive.

## Rollout

`Invar.Rollout` executes a cohort. The driver keeps a counter and reserves one ordinal per member as its call, attempt and instance. Updates reserve from the same counter, so no two executions in a run share a binding. `--devices` gives one session per CUDA device, members are dealt to sessions in execution order, and sessions run concurrently. The inference mode sets the worker lifetime: `serial` and `batch` start one process per session for each cohort, `resident` keeps one process per session for the whole run, and `shared` runs inference and learning in one process.

Each member's adapter load, consumed request and result must match its binding and plan. `Cohort.record` then computes the reward, and `Cohort.admit` requires exactly one result for every member.

## Reward

The reward is a checked semantic program with one sink, `score` / `exact-decimal/v1`, over three sources: the declared answer as an exact rational, the response characters and the truncation flag. A fold over the characters parses the last line as `#### ` followed by a decimal number. The reward is 1 when that number equals the answer exactly and 0 otherwise, and a truncated response scores 0. Every reward the learner consumed can be recomputed from the retained program and inputs.

## Advantage

`Invar.Learn.prepare` compiles the learning program over the batch. Its sources include the policy, learner, reference, algorithm coefficients, trajectories, behavior words, rewards, groups and logical order, and it emits one `update` / `grpo-token-mean/v1` command. Lowering that command into the learner's JSON request also computes each sample's expected advantage (`Invar.Learn.Advantage`) and writes it as `advantage_bits`, one FP32 word per sample.

Within each group, the core sums the binary64 rewards exactly and divides by the group size, subtracts that mean from each reward, computes the population variance the same way, and divides each deviation by the standard deviation plus `delta`. Every operation rounds separately, any value that is not finite fails the request, and each result is rounded once to FP32. The worker computes the same advantages in Python and rejects the request if any word differs from the core's.

## Scalar objective

The objective `grpo-scalar-f32/v1` (`Invar.Learn.Objective`) is defined for each response token from five FP32 words: the current log probability `c`, the proximal log probability `o` from the policy before the update, the behavior log probability `b` from rollout, the reference log probability `q` from the fixed reference adapter, and the advantage `A`. Each operation computes in binary64 and rounds to FP32 at once:

```text
w = exp(o - b)
r = exp(c - o)
d = q - c
k = exp(d)
s = min(r * A, clip(r, 1 - epsilon, 1 + epsilon) * A)
term = -w * s + penalty * (k - d - 1)
```

The weight `w` corrects for any difference between the behavior and proximal policies. The loss is the logical token mean: the sum of terms divided by the total number of response tokens in the batch, so each token counts once however responses are packed. The core also defines each token's cotangent, the derivative of the loss with respect to `c`, and the worker passes these cotangents to the model's autograd to get parameter gradients.

After the update the worker writes `probabilities.json` with every token's five words, terms and cotangents. `Invar.Learn.Probability` recomputes the objective from those words and requires terms, cotangents, the loss and the token count to match bit for bit. It also checks that the behavior words are the ones consumed from rollout and that the advantages are the core's.

## Update execution

`Invar.Learn.Worker.prepare` binds the learning plan to a call, attempt and instance. The worker receives one JSON line holding the invocation, the lowered request and a separate load program, together with the model cache, the current checkpoint, the fixed reference adapter and a new output directory. It reports the learner load, consumption and the result. The core accepts the result only after load and consumption matched the binding, and then verifies the output: `gradients.safetensors` and `learner.pt` by file digest, `adapter.safetensors` by the digest of its tensor contents, and `probabilities.json` by digest and by the recomputation above. Any mismatch stops the cycle before publication.

## Publication

Publication never replaces an existing artifact. The update writes into `staging<N>`, named after the update's ordinal. The core writes the successor policy description, which is the current one with the new adapter digest, to `policy.json` there with an exclusive create. `Invar.Store.publishCheckpoint` syncs the three files and makes the directory visible as `generation<K>`, where `K` counts published generations from 1. `--publication rename` renames the directory with an exclusive rename that fails if the destination exists. `--publication reference` creates `generation<K>` as a symbolic link to the staging directory, for filesystems such as Lustre that refuse exclusive rename.

A checkpoint holds three artifacts with separate roles. `policy.json` is the immutable inference policy description (`invar-policy-v1`): model, revision, adapter, tokenizer, frozen base and model assembly. `adapter.safetensors` holds the adapter tensors that inference loads. `learner.pt` holds the optimizer and random state that only the next update reads.

The driver moves through idle, collecting, updating and publishing. A failure during rollout or planning returns it to idle on the same checkpoint. A failure during the update leaves it unresolved and refuses every later cycle, because the core cannot know what the worker changed. Each cycle reads the published `policy.json` again and requires it to equal the description the driver retained.

## Resident workers

With `--inference-mode resident` and `--learning-mode resident`, the core keeps one inference process per session and one learner process for the whole run. The inference worker loads the model once and switches to each newly published adapter. The learner keeps its model, optimizer, fixed reference and random state in memory. Each group and each update is still admitted separately. After a group, the core sends a release naming exactly that group's loads and transcript digest, and the worker acknowledges once it has freed them. At the end the core sends `close` to every owner and requires an acknowledgement with the number of groups run and a zero exit status.

`--inference-mode shared --learning-mode shared` runs both roles in one process with one session, keeping separate invocation checks for each role. The [MLX backend](../backends/mlx.md) uses this mode on Apple Silicon, and the [vLLM backend](../backends/vllm.md) uses separate resident processes.

## History and comparison

`invar inspect history` admits a complete run, from the initial checkpoint through every published generation to a final standalone inference of the last adapter. `invar compare histories` compares two admitted histories and reports whether tasks, settings, every generation's inputs, inference results, probabilities, gradients and learner states, and the final inference are equal, with schedules reported separately. Two runs with reversed execution order and permuted delivery that compare equal show that scheduling did not reach the published policy. Admission records a SHA-256 of every tensor in each adapter, gradient file and learner checkpoint in the same pass that checks it, and the comparison works from those records, so it reads no artifact a second time.

For each generation the admitted history also reports `learner_engine`, the gap between the learner and the inference engine on the same tokens under the same weights. Each cycle's rollout loads the policy that the update starts from, so the learner's linearized log probability `l`, from the learner graph at the rollout temperature, and the behavior log probability `b` come from one set of weights computed by two implementations. The core already requires the proximal and current words to equal `b`, so the gap it reports is `l - b`; histories recorded before the linearized role existed report `o - b` with the proximal word `o`. `Invar.Learn.Mismatch` reads both words for every token of the admitted probabilities and reports the token count, how many tokens have identical words, and the mean, mean absolute value, median, 90th and 99th percentiles and maximum of the gap in nats. It computes every value exactly from the FP32 words and rounds once to binary64, and each percentile is the smallest absolute gap that at least that share of tokens does not exceed. In an MLX training run recorded before the learner took its probabilities from the engine, reported as `o - b`, 14.8% and 13.2% of tokens were identical in the two generations, with a 99th percentile of 0.27 and 0.28 and a maximum of 1.94.

## Evaluation and inference

`invar evaluate` runs cohorts under one fixed policy with the same rollout driver and reward, and never updates or publishes. `--checkpoint` reads the policy from `policy.json`. For each cohort it prints every sample's reward, response length, truncation flag and binding, with a summary that includes `reward_sum`, `truncated_count` and `zero_variance_groups`. `invar inspect evaluation` admits that output against the frozen workload.

`invar infer` runs one bound inference call from request options, binding options (`--call`, `--attempt`, `--instance`) and either `--checkpoint` or an explicit adapter with its digests. `invar infer batch` runs several calls as one finite group. The use admission of the rollouts in the first acceptance is in [acceptance.md](../results/acceptance.md).
