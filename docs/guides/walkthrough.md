# Walkthrough on Apple Silicon

This walkthrough takes one use decision from declaration to admission on a Mac with the MLX backend. The owner wants to evaluate with the row independent numerics profile in place of the library's stock numerics, and asks two things of it on a small set of arithmetic questions: it must not lose answers, and its results must not change with how requests are batched. Every value below comes from the run recorded on 2026-09-27; the [operating guide](agents.md) explains who decides what.

## Setup

Build Invar with `cabal build exe:invar` and put the binary on `PATH`. Create the Python environment from [the MLX lock file](../../worker/locks/mlx.txt) and download the pinned model, about 15 GB, into a cache directory:

```sh
python -c "from huggingface_hub import snapshot_download; from worker.mlx.model import MODEL, REVISION; snapshot_download(MODEL, revision=REVISION, cache_dir='models')"
```

In the commands below, `$REPO` is the Invar checkout, `$PYTHON` the environment's interpreter and `$MODELS` the cache directory.

## Workload and configurations

`workload.json` holds four questions, each asked with seeds 17 and 18, so eight inputs:

```json
[{"tasks": [
  {"name": "q0/seed17", "group": "q0", "prompt": "A shelf holds 7 rows of 12 books. How many books are on the shelf?\nReason briefly. End your answer with a final line in the form #### <number>.",
   "tokens": 160, "temperature": 0.8, "seed": 17, "answer": "#### 84"},
  …],
  "order": [0, 1, 2, 3, 4, 5, 6, 7], "delivery": [0, 1, 2, 3, 4, 5, 6, 7]}]
```

The two sides differ only in the runtime configuration's `numerics`. `reference.json`:

```json
{"format": "invar-mlx-runtime-v1", "batch_size": 4, "prefill_step": 256, "cache_bytes": 268435456, "numerics": "native"}
```

`candidate.json` is the same with `"numerics": "primary"`.

## Policies

Each side gets its own `policy.json`, sealed from an inference that loaded exactly that side's arithmetic. First read the tokenizer digest:

```sh
tokenizer=$($PYTHON $REPO/entries/mlxoperation.py --cache $MODELS)
```

Then for `reference` and again for `candidate`, initialize the adapter with the side's configuration:

```sh
$PYTHON $REPO/entries/mlxinitialize.py --cache $MODELS --output reference --tokenizer-digest $tokenizer --seed 17 --config reference.json
```

The last record gives the adapter identity as `policy`, and the `profile` record gives the inference `base` and `assembly`. Use those, never the `assembly` of the last record, which belongs to the learner. Run one inference with them and seal the policy from its log:

```sh
invar infer --digest $adapter --tokenizer-digest $tokenizer --base-digest $base --assembly-digest $assembly \
  --prompt "What is 2 + 3? End with a final line #### <number>." --tokens 16 --temperature 0.8 --seed 1 \
  --call 0 --attempt 0 --instance 0 \
  --python $PYTHON --worker $REPO/entries/mlxinfer.py --worker-config reference.json --cache $MODELS \
  --adapter reference/adapter.safetensors > reference/inference.jsonl
invar policy --digest $adapter --tokenizer-digest $tokenizer --base-digest $base --assembly-digest $assembly \
  --prompt "What is 2 + 3? End with a final line #### <number>." --tokens 16 --temperature 0.8 --seed 1 \
  --call 0 --attempt 0 --instance 0 --checkpoint reference --log reference/inference.jsonl --exit-code 0
```

Both sides share the adapter `65b96ed1…`. Their assemblies differ: `7a69568a…` for the stock numerics and `9f1e1c3d…` for the row independent ones.

## Declaration and execution plan

The declaration records the owner's decisions. The budgets and rationales are theirs:

```json
{"format": "invar-use-declaration-v1",
 "artifacts": {"workload": "workload.json", "reference_policy": "reference/policy.json", "candidate_policy": "candidate/policy.json"},
 "decisions": {
  "purpose": "Row-invariant MLX numerics replace the stock numerics for evaluation on this finite exact-decimal set",
  "scoring": {"method": "exact-decimal"},
  "context_tokens": 512,
  "quality": {
   "reference_ceiling": {"limit": "1/2", "rationale": "the stock numerics must already answer at least half of these questions for the comparison to mean anything"},
   "regression_ceiling": {"limit": "1/8", "rationale": "losing more than one of the eight answers would make evaluation results depend on the numerics"},
   "standard": {"kind": "finite_domain"}},
  "requirements": {
   "invariance": [{"relation": {"kind": "behavior-bits"}, "executions": 2, "rationale": "evaluation results must not depend on how requests are batched"}],
   "numerical": []},
  "protocols": {"freeze": "…", "isolation": "No acceptance result was used to choose, change or rerun the candidate", "selection": "…"},
  "reliance": [{"premise": "ContractFrozen", "authority": "walkthrough owner", "basis": ["basis/record.md"]}, …]}}
```

`execution.json` says how to run it. Paths resolve from the file's directory. The paired runs take the inputs in declared order, four per batch. The one repeat rotates the order by two, so every input meets different batch neighbours:

```json
{"format": "invar-use-execution-v1", "backend": "mlx", "python": "env/bin/python", "cache": "models",
 "sides": {"reference": {"worker": "entries/mlxbatch.py", "configuration": "reference.json", "adapter": "reference/adapter.safetensors"},
           "candidate": {"worker": "entries/mlxbatch.py", "configuration": "candidate.json", "adapter": "candidate/adapter.safetensors"}},
 "paired": {"order": "declared", "group_size": 4},
 "repeats": [{"order": "rotated", "offset": 2, "group_size": 4}],
 "collection": {"scores": false, "full_vocabulary_steps": []}}
```

## Prepare

```sh
invar use prepare --declaration declaration.json --execution execution.json --output contract.json
```

With an empty `reliance` list, `prepare` refuses and lists the eleven premises this contract raises, from `ParameterMeaning` to `SelectionControl`, each as `missing-reliance`. Once the owner has named an authority and a basis for each, it prints `"status": "prepared"`. It reports 8 inputs in 4 units, two seeds per unit, and 2 candidate executions, and it restates each decision under `owner_review`, for example "The candidate may raise the mean task loss by at most 1/8 over the reference, because losing more than one of the eight answers would make evaluation results depend on the numerics."

`invar use plan --contract contract.json --units 4` answers `"standard": "finite_domain"` with no bound: a finite domain is decided by its exact mean, so there is no sample size to plan.

## Run

```sh
invar use run --contract contract.json --execution execution.json --output run
```

It checks the adapters against the two policies, then runs six finite batches, two per side and two for the repeat. Each batch loads the model once, and the whole run took 293 seconds. The output is `"status": "recorded"`, and `run` holds `contract.json`, `plan.json`, `records.json`, `runs.json` and every batch log under `logs`.

## Admit

```sh
cd run
invar use admit --contract contract.json --runs runs.json
```

`decision.status` is `admitted_under_declared_reliance`. Both sides answered all eight questions, so the reference loss, the candidate loss and the increase are all 0, within the ceilings of 1/2 and 1/8. The candidate's behavior bits were identical in both schedules for all eight inputs, which meets the invariance requirement. The decision lists 117 remaining conditions, each a premise that rests on the owner's declared reliance, attached to the input, side or batch it covers.

The observation also shows how the two numerics differ, as diagnostics that do not enter the decision. The stock and row independent profiles produced the same tokens for seven of the eight inputs, and identical probability bits for none. The largest log ratio over a shared prefix was 0.25.

Running `invar use admit` again on the same directory prints the same bytes.
