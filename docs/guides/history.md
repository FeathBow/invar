# Runtime histories

`invar train --staleness d` runs rollout and learning concurrently: update u learns from rollouts of version max(0, u − d) while the next rollouts run. Such a run keeps its evidence in its output directory. This guide covers what that directory holds, how to resume a run, how to admit it as a history and compare it with another run, and which files to keep.

The runtime runs with the `serial`, `batch` or `resident` worker modes, and with `shared` execution at `--staleness 0`, where one process carries both roles. With `--staleness 0` every update learns from rollouts of the policy it starts from, as in the synchronous loop.

## The output directory

`--output` names a new directory. It holds:

- `journal.jsonl`, the run's declaration followed by every event in order. The declaration records the directory the run started in, the training arguments and the workload.
- `transcripts/<N>.jsonl`, everything process N wrote, one file for each inference, learner or shared process.
- `staging<N>`, the files one update wrote: its adapter, learner state, gradients, probabilities and successor policy description.
- `generation<K>`, the K-th published checkpoint. With `--publication reference` it is a symbolic link to a staging directory in the same output directory.

## Resuming

`invar train --resume DIRECTORY` continues an interrupted run and takes no other option. It replays the journal against the transcripts and generations, journals a restart and continues with the updates that are not yet published. An update without a published generation runs again from its rollouts; a generation published before the journal recorded it is confirmed by the restart. When the replay refuses the directory, the journal stays as it was.

A running run holds its journal, so a resume or an inspection of a run that is still running is refused.

## Inspecting a run

`invar inspect history --run DIRECTORY` admits a finished run. The run's declaration supplies the training arguments, the workload, the initial checkpoint and the reference. The other options give the evidence from outside the run: the initial state source and RNG profile, the profile mode, the codec, and a final independent inference of the last generation. Run that inference with `invar infer --checkpoint DIRECTORY/generation<K>` under a call, attempt and instance number above every one the run journaled, and pass its request, binding, stdout and exit status as the `--final-` options. In the example `$REPO` is the Invar checkout and `$PYTHON` the worker environment's interpreter:

```sh
invar inspect history --run run --python "$PYTHON" --codec "$REPO/entries/codec.py" \
  --cuda-rng-vectors 0 --initial-source provided --profile-mode unreported \
  --final-digest $ADAPTER --final-tokenizer-digest $TOKENIZER --final-base-digest $BASE --final-assembly-digest $ASSEMBLY \
  --final-prompt "$PROMPT" --final-tokens 4 --final-temperature 0.8 --final-seed 17 \
  --final-call 1000 --final-attempt 1000 --final-instance 1000 \
  --final-log final.jsonl --final-exit-code 0
```

The admission replays the journal as recorded. It requires every declared update to be committed, every process to have ended and the generations in the directory to be exactly the committed ones. It then checks each generation as it does for a training log: publication, artifacts, learner state, optimizer steps, profiles and model agreement. The output reports the staleness, each update's requests and behavior version, every attempt with its process, the calls it consumed and its outcome (committed, concluded without publication or incomplete), the observations of each restart, the update schedule and the generations. Inspection opens the journal read-only and leaves the directory as it found it.

## Comparing runs

`invar compare histories` takes each side's history options with a `--left-` or `--right-` prefix, so either side can be a run directory (`--left-run`) or a training log. The comparison is equal when tasks, settings, the update schedule, the initial state, every generation's numbers and the final inference agree. For each update the schedule lists its staleness, the behavior version of its samples, the logical sample order and the optimizer steps. A synchronous run and a run with `--staleness 0` of the same workload and settings can therefore compare equal, and runs with different staleness report it in `schedule.differences`. The `execution` field reports, for each side, what it declared about sessions, worker modes, dispatch order, delivery and the final binding, and what a run's journal records about attempts, processes and restarts.

## What to keep

Inspection reads the following, and a resume reads the first two:

- the output directory as a whole: the journal, every transcript, every staging directory and every generation;
- the initial checkpoint directory and the reference adapter the arguments name;
- the stdout and exit status of the final independent inference, and of the initializer when the initial state came from one.

Relative paths in the arguments are read from the directory the run started in, so that directory and the files it leads to stay in place. The output directory stays at the path its declaration names, because a resume and an inspection compare that path with the directory they open. A generation link names its staging directory, so both stay in the same output directory. The initial checkpoint and the reference keep the files whose digests the arguments declare.
