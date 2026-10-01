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

`Invar.Learn.prepare` compiles the learning program over the batch. Its sources include the policy, learner, reference, algorithm coefficients, the update's schedule, trajectories with the version and behavior policy that generated each one, behavior words, rewards, groups and logical order, and it emits one `update` / `grpo-token-mean/v1` command. Every sample must come from version max(0, u − d) of update u under the declared staleness d; the synchronous loop runs with d = 0, so each update learns from rollouts of the policy it starts from. Lowering that command into the learner's JSON request also computes each sample's expected advantage (`Invar.Learn.Advantage`) and writes it as `advantage_bits`, one FP32 word per sample.

Within each group, the core sums the binary64 rewards exactly and divides by the group size, subtracts that mean from each reward, computes the population variance the same way, and divides each deviation by the standard deviation plus `delta`. Every operation rounds separately, any value that is not finite fails the request, and each result is rounded once to FP32. The worker reads these words as given.

## Scalar objective

The objective `grpo-scalar-f32/v1` (`Invar.Learn.Objective`) is defined for each response token from five FP32 words: the current log probability `c` from the learner at the weights of the current optimizer step, the proximal log probability `o` from the learner before the first optimizer step, the behavior log probability `b` from rollout, the reference log probability `q` from the fixed reference adapter, and the advantage `A`. Each operation computes in binary64 and rounds to FP32 at once:

```text
w = exp(o - b)
r = exp(c - o)
d = q - c
k = exp(d)
s = min(r * A, clip(r, 1 - epsilon, 1 + epsilon) * A)
term = -w * s + penalty * (k - d - 1)
```

The weight `w` corrects for the difference between the behavior and proximal policies. The update runs the optimizer steps declared with `invar train --steps`, which split the logical order into consecutive mini-batches; the default is one step over the whole batch. The loss of a step is its logical token mean: the sum of its terms divided by the number of response tokens in that step, so each token counts once however responses are packed. The core defines each token's cotangent, the derivative of that loss with respect to `c`, and the derivative of the reward part alone.

The learner computes `c` and the core computes everything that depends on it. Before the first step the learner reports the proximal words of every sample outside the first step. Within a step it takes the samples in order, runs the forward pass, reports the current words together with their digest and the state digest of its weights, and receives the two cotangent vectors the core computed for exactly that report. It applies them through one vector-Jacobian product and adds the result to the gradients before it moves to the next sample, so it holds one sample's graph at a time. For samples in the first step, the current words are also the proximal words. After the last sample the learner runs AdamW and reports the state before and after with the digests of the cotangent tensors it used, in order. The core requires each step to start from the state the previous one ended in and the last to end at the adapter the update returns.

After the update the worker writes `probabilities.json` with each sample's proximal words and its current words in every step. `Invar.Learn.Probability` requires them to equal the words the core checked during the exchange.

## Update execution

`Invar.Learn.Worker.prepare` binds the learning plan to a call, attempt and instance. The worker receives one JSON line holding the invocation, the lowered request and a separate load program, together with the model cache, the current checkpoint, the fixed reference adapter and a new output directory. It reports the learner load and consumption, exchanges the optimizer steps described above, and reports the result. The core accepts the result only after load, consumption and every step matched the binding, and then verifies the output: `gradients.safetensors`, which holds the first step's gradients before AdamW, and `learner.pt` by file digest, `adapter.safetensors` by the digest of its tensor contents, and `probabilities.json` by digest and against the exchange. The gradient norms in the update summary cover the same first step. Any mismatch stops the cycle before publication.

## Publication

Publication never replaces an existing artifact. The update writes into `staging<N>`, named after the update's ordinal. The core writes the successor policy description, which is the current one with the new adapter digest, to `policy.json` there with an exclusive create. `Invar.Store.publishCheckpoint` syncs the three files and makes the directory visible as `generation<K>`, where `K` counts published generations from 1. `--publication rename` renames the directory with an exclusive rename that fails if the destination exists. `--publication reference` creates `generation<K>` as a symbolic link to the staging directory, for filesystems such as Lustre that refuse exclusive rename.

A checkpoint holds three artifacts with separate roles. `policy.json` is the immutable inference policy description (`invar-policy-v1`): model, revision, adapter, tokenizer, frozen base and model assembly. `adapter.safetensors` holds the adapter tensors that inference loads. `learner.pt` holds the optimizer and random state that only the next update reads.

The driver moves through idle, collecting, updating and publishing. A failure during rollout or planning returns it to idle on the same checkpoint. A failure during the update leaves it unresolved and refuses every later cycle, because the core cannot know what the worker changed. Each cycle reads the published `policy.json` again and requires it to equal the description the driver retained.

## Resident workers

With `--inference-mode resident` and `--learning-mode resident`, the core keeps one inference process per session and one learner process for the whole run. The inference worker loads the model once and switches to each newly published adapter. The learner keeps its model, optimizer, fixed reference and random state in memory. Each group and each update is still admitted separately. After a group, the core sends a release naming exactly that group's loads and transcript digest, and the worker acknowledges once it has freed them. At the end the core sends `close` to every owner and requires an acknowledgement with the number of groups run and a zero exit status.

`--inference-mode shared --learning-mode shared` runs both roles in one process with one session, keeping separate invocation checks for each role. The [MLX backend](../backends/mlx.md) uses this mode on Apple Silicon, and the [vLLM backend](../backends/vllm.md) uses separate resident processes.

## History and comparison

`invar inspect history` admits a complete run, from the initial checkpoint through every published generation to a final standalone inference of the last adapter. `invar compare histories` compares two admitted histories and reports whether tasks, settings, every generation's inputs, inference results, probabilities, gradients and learner states, and the final inference are equal, with schedules reported separately. Two runs with reversed execution order and permuted delivery that compare equal show that scheduling did not reach the published policy. Admission records a SHA-256 of every tensor in each adapter, gradient file and learner checkpoint in the same pass that checks it, and the comparison works from those records, so it reads no artifact a second time.

For each generation the admitted history also reports `learner_engine`, the gap between the learner and the inference engine on the same tokens under the same weights. Each cycle's rollout loads the policy that the update starts from, so the learner's proximal log probability `o`, from the learner graph at the rollout temperature before the first optimizer step, and the behavior log probability `b` come from one set of weights computed by two implementations. The gap it reports is `o - b`. `Invar.Learn.Mismatch` reads both words for every token of the admitted probabilities and reports the token count, how many tokens have identical words, and the mean, mean absolute value, median, 90th and 99th percentiles and maximum of the gap in nats. It computes every value exactly from the FP32 words and rounds once to binary64, and each percentile is the smallest absolute gap that at least that share of tokens does not exceed. When the update takes its reference log probabilities from the learner (`--reference-source learner`), the admitted history also reports `reference_engine`: the same summary between the engine's retained reference scores and the learner's own reference words on the same tokens, in the direction learner minus engine. It is reported only when the engine kept scores and the learner produced words for the same tokens, which is the only case where one quantity has two implementations. The engine's scores stay in the declaration whether or not the update uses them.

## Evaluation and inference

`invar evaluate` runs cohorts under one fixed policy with the same rollout driver and reward, and never updates or publishes. `--checkpoint` reads the policy from `policy.json`. For each cohort it prints every sample's reward, response length, truncation flag and binding, with a summary that includes `reward_sum`, `truncated_count` and `zero_variance_groups`. `invar inspect evaluation` admits that output against the frozen workload.

`invar infer` runs one bound inference call from request options, binding options (`--call`, `--attempt`, `--instance`) and either `--checkpoint` or an explicit adapter with its digests. `invar infer batch` runs several calls as one finite group. The use admission of the rollouts in the first acceptance is in [acceptance.md](../results/acceptance.md).
