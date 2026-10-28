# Runtime histories

`invar train` runs rollout and learning through the event runtime: update u learns from rollouts of version max(0, u − d) while later rollouts run, where d is `--staleness` and 0 when the option is omitted. With d = 0 every update learns from rollouts of the policy it starts from. The run keeps its evidence in its output directory and prints only a summary on standard output: a `published` record for each generation and a `resumed` record when it resumes. The full history is read from the directory with `invar inspect history --run`. This guide covers what that directory holds, how to resume a run, how to admit it as a history and compare it with another run, and which files to keep.

The runtime runs with the `serial`, `batch` or `resident` worker modes, and with `shared` execution, where one process carries both roles at staleness 0. `invar inspect history --log` admits a training log: the standard output of a run that kept no journal and wrote every worker record there, one cycle after another.

## The output directory

`--output` names a new directory. It holds:

- `journal.jsonl`, the run's declaration followed by every event in order. The declaration records the directory the run started in, the training arguments and the workload.
- `transcripts/<N>.jsonl`, everything process N wrote, one file for each inference, learner, shared or teacher process. A distillation run starts one teacher process for each rollout it scores, or one for the whole run with `--teacher-mode resident`.
- `staging<N>`, the files one update wrote: its adapter, learner state, gradients, probabilities and successor policy description.
- `generation<K>`, the K-th published checkpoint. With `--publication reference` it is a symbolic link to a staging directory in the same output directory.

## Resuming

`invar train --resume DIRECTORY` continues an interrupted run of the event runtime and takes no other option. It replays the journal against the transcripts and generations, journals a restart and continues with the updates that are not yet published. An update without a published generation runs again from its rollouts; a generation published before the journal recorded it is confirmed by the restart. When the replay refuses the directory, the journal stays as it was.

A running run holds its journal, so a resume or an inspection of a run that is still running is refused. A directory without `journal.jsonl`, such as the output of a run recorded as a training log, is refused because it has no run journal.

## Inspecting a run

`invar inspect history --run DIRECTORY` admits a finished run. The run's declaration supplies the training arguments, the workload, the initial checkpoint and the reference. The other options give the evidence from outside the run: the initial state source and RNG profile, the profile mode, the codec, and a final independent inference of the last generation. Run that inference with `invar infer --checkpoint DIRECTORY/generation<K>` under a call, attempt and instance number above every one the run journaled, and pass its request, binding, stdout and exit status as the `--final-` options. The four `--final-` digests are the `adapter`, `tokenizer`, `base` and `assembly` fields that `invar policy inspect --checkpoint DIRECTORY/generation<K>` prints. In the example `$REPO` is the Invar checkout and `$PYTHON` the worker environment's interpreter:

```sh
invar inspect history --run run --python "$PYTHON" --codec "$REPO/entries/codec.py" \
  --cuda-rng-vectors 0 --initial-source provided --profile-mode unreported \
  --final-digest $ADAPTER --final-tokenizer-digest $TOKENIZER --final-base-digest $BASE --final-assembly-digest $ASSEMBLY \
  --final-prompt "$PROMPT" --final-tokens 4 --final-temperature 0.8 --final-seed 17 \
  --final-call 1000 --final-attempt 1000 --final-instance 1000 \
  --final-log final.jsonl --final-exit-code 0
```

The admission replays the journal as recorded. It requires every declared update to be committed, every process to have ended and the generations in the directory to be exactly the committed ones. It then checks each generation as it does for a training log: publication, artifacts, learner state, optimizer steps, profiles and model agreement. The output reports the staleness, each update's requests and behavior version, every attempt with its process, the calls it consumed and its outcome (committed, concluded without publication or incomplete), the observations of each restart, the update schedule and the generations. Inspection opens the journal read-only and leaves the directory as it found it. Each generation's `learner_engine` and optional `reference_engine` are computed from path observations of the admitted request and probability artifact, not from a separate summary: each observation names the token path, distribution rule, forward weights, implementation and per-token words. `learner_engine.path_log_ratio` is the exact learner-minus-engine sum over the response tokens. Two observations are comparable only when their path and distribution rule agree and their forward-weight identities are equal or bound by a checked correspondence; assemblies may differ. Different forward weights are a policy difference, not an implementation difference, and a contract requirement whose evidence has only such pairs is undetermined.

A use contract can name a training–inference relation in `criterion.numerical`: `training-path-words` requires identical FP32 words on every response token, `training-path-token-log-gap` bounds the largest per-token absolute log-probability difference in nats, and `training-path-log-ratio` bounds the absolute sum of that difference over the path. The relation's `learner` field names which side is the learner. `invar use inspect` and `invar use admit` read the training history with the same `--history-run` or `--history-log` and `--history-final-` options as `invar inspect history`, attach its path observations to the contract inputs, and decide admitted, violated or undetermined from the retained records. The check uses each update's proximal observation and, when the learner scored the reference, its reference observation; current observations are reported but not claimed, because after the first optimizer step they no longer share the behavior policy's weights. The contract's per-token budget is fixed before acceptance and describes only the recorded implementations.

## Comparing runs

`invar compare histories` takes each side's history options with a `--left-` or `--right-` prefix, so either side can be a run directory (`--left-run`) or a training log. The comparison is equal when tasks, settings, the update schedule, the initial state, every generation's numbers and the final inference agree. For each update the schedule lists its staleness, the behavior version of its samples, the logical sample order and the optimizer steps. A training log and a runtime run at staleness zero of the same workload and settings can therefore compare equal, and runs with different staleness report it in `schedule.differences`. The `execution` field reports, for each side, what it declared about sessions, worker modes, dispatch order, delivery and the final binding, and what a run's journal records about attempts, processes and restarts.

This compares a run at staleness 0 in `run0` with a run at staleness 1 in `run1`, each with the final inference of its own last generation:

```sh
invar compare histories --python "$PYTHON" --codec "$REPO/entries/codec.py" \
  --left-run run0 --left-cuda-rng-vectors 0 --left-initial-source provided --left-profile-mode unreported \
  --left-final-digest $ADAPTER0 --left-final-tokenizer-digest $TOKENIZER --left-final-base-digest $BASE --left-final-assembly-digest $ASSEMBLY \
  --left-final-prompt "$PROMPT" --left-final-tokens 4 --left-final-temperature 0.8 --left-final-seed 17 \
  --left-final-call 1000 --left-final-attempt 1000 --left-final-instance 1000 \
  --left-final-log final0.jsonl --left-final-exit-code 0 \
  --right-run run1 --right-cuda-rng-vectors 0 --right-initial-source provided --right-profile-mode unreported \
  --right-final-digest $ADAPTER1 --right-final-tokenizer-digest $TOKENIZER --right-final-base-digest $BASE --right-final-assembly-digest $ASSEMBLY \
  --right-final-prompt "$PROMPT" --right-final-tokens 4 --right-final-temperature 0.8 --right-final-seed 17 \
  --right-final-call 1000 --right-final-attempt 1000 --right-final-instance 1000 \
  --right-final-log final1.jsonl --right-final-exit-code 0
```

A training log takes `--left-log` with the training arguments, the workload, the initial checkpoint and the reference in place of `--left-run`, as `invar inspect history --log` does.

## What to keep

Inspection reads the following, and a resume reads the first two:

- the output directory as a whole: the journal, every transcript, every staging directory and every generation;
- the initial checkpoint directory and the reference adapter the arguments name;
- the stdout and exit status of the final independent inference, and of the initializer when the initial state came from one.

Relative paths in the arguments are read from the directory the run started in, so that directory and the files it leads to stay in place. The output directory stays at the path its declaration names, because a resume and an inspection compare that path with the directory they open. A generation link names its staging directory, so both stay in the same output directory. The initial checkpoint and the reference keep the files whose digests the arguments declare.
