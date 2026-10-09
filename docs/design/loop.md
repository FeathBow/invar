# The owning training loop

The Haskell core owns the training loop. Each cycle it chooses the policy, declares the cohort, dispatches rollouts, scores the responses, derives the advantages, binds one update, checks the update's artifacts and publishes the next checkpoint. Python workers do the numerical work in between: they load models, generate, compute probabilities and gradients, and step the optimizer. Every worker report passes through the invocation binding described in [contract.md](contract.md), so a run's history is a sequence of checked facts that can be admitted and compared later.

`invar train` runs this loop through one of two entry points: the event runtime by default, and the lockstep driver in `Invar.Loop` for a shared run. One cycle runs this sequence:

1. Read the current checkpoint's `policy.json` and bind every task to that policy description.
2. Run the cohort through the rollout driver and record each result with its reward.
3. Compile the learning program over the admitted batch, which fixes rewards, groups and advantages.
4. Bind one update to a fresh call, attempt and instance and run it on the learner.
5. Verify the gradients, probabilities, adapter and learner state the worker wrote.
6. Publish the checkpoint under a new name and make it current for the next cycle.

## Entry points

`invar train` runs a non-shared run through the event runtime, at staleness zero unless `--staleness d` gives it a positive staleness. A shared run uses the lockstep driver in `Invar.Loop` and reaches the runtime only by declaring `--staleness 0`; the cross-role agreement that shared execution needs is enforced there. `invar evaluate` uses the rollout driver alone and publishes nothing.

The event runtime runs the same rollout and learning workers and the same checks as the driver, and adds a journal and a recovery path. It keeps one thread for collection and one for updates, so with a positive staleness update u learns from rollouts of version max(0, u − d) while later rollouts may already run; a staler update can only consume a version that is already published. It writes `journal.jsonl` and one `transcripts/<N>.jsonl` for each inference, learner or shared process into `--output`. `invar train --resume` replays that journal against those transcripts and the published generations to continue an interrupted run, and `invar inspect history --run` admits a finished run from the same directory. A shared run through the runtime is the shared execution described under Resident workers.


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

## Request and step contract

A field of the learner's request is declared once and checked once. `Invar.Learn.Program` declares it among the program's sources and in the sink type, and `Invar.Learn.prepare` supplies its value from the settings and the batch. `Invar.Learn.Wire` lowers the emitted command into the JSON request. `Invar.Learn.Request` is the one parser of that request: it refuses missing, unknown and invalid fields, and gives every consumer in the core a checked request with the exchange plan, the logical order and the learner materialization. The live exchange, inspection, comparison and history all read that checked request. The worker checks its own copy in `worker/cohort.py` before it loads a model; that check belongs to the worker.

Step reports are defined in `Invar.Learn.Step`: the stages `proximal`, `reference`, `current` and `applied`, how each is decoded, and how each advances an `Invar.Learn.Stream`. The stream checks their content against the plan, including order, lengths, observations, the state chain and the declared reference source, and computes each reply's cotangents. The engine's reference words for a sample are its retained reference scores, or its behavior words when the reference is the behavior policy; `Invar.Learn.Stream.engineWords` holds that rule for both the objective and the history's diagnostic. The probability artifact, read by `Invar.Learn.Probability`, embeds the checked request and must repeat the reported words.

A new request field therefore changes the program declaration, the settings that supply it, the lowering, the request parser and the worker's check. A new step stage changes `Invar.Learn.Step` and the worker that reports it.

## Publication

Publication never replaces an existing artifact. The update writes into `staging<N>`, named after the update's ordinal. The core writes the successor policy description, which is the current one with the new adapter digest, to `policy.json` there with an exclusive create. `Invar.Store.publishCheckpoint` syncs the three files and makes the directory visible as `generation<K>`, where `K` counts published generations from 1. `--publication rename` renames the directory with an exclusive rename that fails if the destination exists. `--publication reference` creates `generation<K>` as a symbolic link to the staging directory, for filesystems such as Lustre that refuse exclusive rename.

A checkpoint holds three artifacts with separate roles. `policy.json` is the immutable inference policy description (`invar-policy-v1`): model, revision, adapter, tokenizer, frozen base and model assembly. `adapter.safetensors` holds the adapter tensors that inference loads. `learner.pt` holds the optimizer and random state that only the next update reads.

In the lockstep driver the run moves through idle, collecting, updating and publishing. A failure during rollout or planning returns it to idle on the same checkpoint. A failure during the update leaves it unresolved and refuses every later cycle, because the core cannot know what the worker changed. Each cycle reads the published `policy.json` again and requires it to equal the description the driver retained. The event runtime reports the same distinction: a failure that ends the run names the update whose worker may have run, and `invar inspect history` reads each attempt's outcome — committed, concluded without publication, or incomplete — from the journal.

## Resident workers

With `--inference-mode resident` and `--learning-mode resident`, the core keeps one inference process per session and one learner process for the whole run. The inference worker loads the model once and switches to each newly published adapter. The learner keeps its model, optimizer, fixed reference and random state in memory. Each group and each update is still admitted separately. After a group, the core sends a release naming exactly that group's loads and transcript digest, and the worker acknowledges once it has freed them. At the end the core sends `close` to every owner and requires an acknowledgement with the number of groups run and a zero exit status.

`--inference-mode shared --learning-mode shared` runs both roles in one process with one session, keeping separate invocation checks for each role. Without `--staleness` a shared run runs on the lockstep driver, which owns the cross-role agreement; `--staleness 0` moves it to the event runtime, which runs the same process: the journal reserves it once under the role `shared`, its transcript holds both roles' records, and each rollout waits for the previous publication. Shared execution with a positive staleness is refused. The [MLX backend](../backends/mlx.md) uses this mode on Apple Silicon, and the [vLLM backend](../backends/vllm.md) uses separate resident processes.

## History and comparison

`invar inspect history` admits a complete run, from the initial checkpoint through every published generation to a final standalone inference of the last adapter. It replays every worker session through the same machine as the online run, under the policy description each generation ran with (the initial checkpoint's, then the successor of each publication) and the reference the run scored with. A finite session's results count only after the whole history, every resident close and the run's exit status have been checked. A run of the event runtime keeps its evidence in its output directory, and `invar inspect history --run` admits it from there by replaying its journal against every process transcript and published generation; [history.md](../guides/history.md) describes the commands and what a run must keep.

`invar compare histories` compares two admitted histories of either kind and reports whether tasks, settings, the update schedule, every generation's inputs, inference results, probabilities, gradients and learner states, and the final inference are equal. The update schedule is read from each admitted update request: the update, its staleness, the behavior version of its samples, the logical sample order and the optimizer steps. A synchronous run has staleness zero, so it can equal a runtime run with staleness zero. Beside that conclusion the comparison reports each side's execution: the declared session count, worker modes, each cycle's dispatch order and delivery and the final inference binding, and for a runtime run the attempts, processes and restarts its journal records. When an execution choice changes a probability, gradient or state, the numerical comparisons show it. Two runs with reversed execution order, permuted delivery or a different number of sessions that compare equal show that the execution did not reach the published policy. Admission records a SHA-256 of every tensor in each adapter, gradient file and learner checkpoint in the same pass that checks it, and the comparison works from those records, so it reads no artifact a second time.

For each generation the admitted history also reports `learner_engine`, the gap `o - b` between the learner's proximal log probability `o`, from the learner graph at the rollout temperature before the first optimizer step, and the behavior log probability `b` the inference engine reported for the same tokens. The learner computes `o` with the weights the update starts from. When the samples come from that same policy, which holds for every update of a synchronous run and for the first update of any run, `o` and `b` come from one set of weights computed by two implementations. With a positive staleness d, update u learns from rollouts of version max(0, u − d), so for u > 0 the gap also contains the change between those two policies. `Invar.Learn.Mismatch` reads both words for every token of the admitted probabilities and reports the token count, how many tokens have identical words, and the mean, mean absolute value, median, 90th and 99th percentiles and maximum of the gap in nats. It computes every value exactly from the FP32 words and rounds once to binary64, and each percentile is the smallest absolute gap that at least that share of tokens does not exceed. When the update takes its reference log probabilities from the learner (`--reference-source learner`), the admitted history also reports `reference_engine`: the same summary between the engine's retained reference scores and the learner's own reference words on the same tokens, in the direction learner minus engine. It is reported for every sample whose reference words the learner produced, which is the only case where one quantity has two implementations; the engine side is its retained reference scores, or its behavior words when it kept none because the reference adapter is the rollout policy, which is how the objective itself takes them. The engine's scores stay in the declaration whether or not the update uses them.

## Evaluation and inference

`invar evaluate` runs cohorts under one fixed policy with the same rollout driver and reward, and never updates or publishes. `--checkpoint` reads the policy from `policy.json`. For each cohort it prints every sample's reward, response length, truncation flag and binding, with a summary that includes `reward_sum`, `truncated_count` and `zero_variance_groups`. `invar inspect evaluation` admits that output against the frozen workload. It takes the worker mode and the checkpoint or policy the evaluation ran with, replays every worker session, and joins each sample to the inference of its task, whose reward, response length, truncation flag and seed it must repeat.

`invar infer` runs one bound inference call from request options, binding options (`--call`, `--attempt`, `--instance`) and either `--checkpoint` or an explicit adapter with its digests. `invar infer batch` runs several calls as one finite group. The use admission of the rollouts in the first acceptance is in [acceptance.md](../results/acceptance.md).
