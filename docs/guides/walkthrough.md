# Walkthrough on Apple Silicon

This walkthrough takes one use decision from declaration to admission on a Mac with the MLX backend. The owner wants to evaluate with the row independent numerics profile in place of the library's stock numerics, and asks two things of it on a small set of arithmetic questions: it must not lose answers, and its results must not change with how requests are batched. Every value below comes from one recorded run; the [operating guide](agents.md) explains who decides what.

The files the owner wrote are in [walkthrough](walkthrough): the workload, the two runtime configurations, the declaration and the record its reliance points to. They hold that owner's decisions. For your own use, the owner states the purpose, budgets, rationales, protocols and reliance again; an operator does not reuse them.

## Setup

Build Invar with `cabal build exe:invar` and put the binary on `PATH`. Create a Python environment from [the MLX lock file](../../worker/locks/mlx.txt). Set three variables and copy the example into a fresh working directory:

```sh
REPO=$HOME/invar                  # the Invar checkout
PYTHON=$HOME/invar-mlx/bin/python # the environment's interpreter
MODELS=$HOME/invar-models         # the model cache
cp -R $REPO/docs/guides/walkthrough work
cd work
```

Download the pinned model, about 15 GB:

```sh
PYTHONPATH=$REPO $PYTHON -c "from huggingface_hub import snapshot_download; from worker.mlx.model import MODEL, REVISION; snapshot_download(MODEL, revision=REVISION, cache_dir='$MODELS')"
```

## Workload and configurations

`workload.json` holds four questions, each asked with seeds 17 and 18, so eight inputs. The first reads:

```json
{"name": "q0/seed17", "group": "q0", "tokens": 160, "temperature": 0.8, "seed": 17, "answer": "#### 84",
 "prompt": "A shelf holds 7 rows of 12 books. How many books are on the shelf?\nReason briefly. End your answer with a final line in the form #### <number>."}
```

The two sides differ only in the runtime configuration's `numerics`: `reference.json` selects `native`, the library's projection, LoRA and attention, and `candidate.json` selects `primary`, the row independent profile. Both use batches of 4.

## Policies

Each side gets its own `policy.json`, written from an inference that loaded exactly that side's arithmetic. The initializer reports two sets of identities. Its final `initial` record carries the adapter identity as `policy` and the learner's `base` and `assembly`; its `profile` record carries the inference `base` and `assembly` under `inference`. Inference and `policy.json` use the inference ones.

```sh
tokenizer=$($PYTHON $REPO/entries/mlxoperation.py --cache $MODELS)
for side in reference candidate; do
  $PYTHON $REPO/entries/mlxinitialize.py --cache $MODELS --output $side --tokenizer-digest $tokenizer \
    --seed 17 --config $side.json > $side-initialize.jsonl
  adapter=$(jq -r 'select(.stage == "initial") | .policy' $side-initialize.jsonl)
  base=$(jq -r 'select(.stage == "profile") | .inference.base' $side-initialize.jsonl)
  assembly=$(jq -r 'select(.stage == "profile") | .inference.assembly' $side-initialize.jsonl)
  request=(--digest $adapter --tokenizer-digest $tokenizer --base-digest $base --assembly-digest $assembly
           --prompt "What is 2 + 3? End with a final line #### <number>." --tokens 16 --temperature 0.8 --seed 1
           --call 0 --attempt 0 --instance 0)
  invar infer "${request[@]}" --python $PYTHON --worker $REPO/entries/mlxinfer.py --worker-config $side.json \
    --cache $MODELS --adapter $side/adapter.safetensors > $side/inference.jsonl
  invar policy "${request[@]}" --checkpoint $side --log $side/inference.jsonl --exit-code 0
done
```

Both sides share the initialized adapter and have different inference assemblies for the stock and row independent numerics. The commands above read those identities from each side's initialization records.

## Declaration and execution plan

`declaration.json` records the owner's decisions: the purpose, exact-decimal scoring, a context of 512 tokens, a reference ceiling of 1/2 and a regression ceiling of 1/8 over the finite domain, and one invariance requirement that the candidate's behavior bits stay the same across 2 executions. Every budget and requirement carries the owner's rationale, and each of the eleven premises names the owner as authority with `basis/record.md` as basis.

The execution plan says how to run it. The paired runs take the inputs in declared order, four per batch; the one repeat rotates the order by two, so every input meets different batch neighbours:

```sh
cat > execution.json <<EOF
{"format": "invar-use-execution-v1", "backend": "mlx", "python": "$PYTHON", "cache": "$MODELS",
 "sides": {"reference": {"worker": "$REPO/entries/mlxbatch.py", "configuration": "reference.json", "adapter": "reference/adapter.safetensors"},
           "candidate": {"worker": "$REPO/entries/mlxbatch.py", "configuration": "candidate.json", "adapter": "candidate/adapter.safetensors"}},
 "paired": {"order": "declared", "group_size": 4},
 "repeats": [{"order": "rotated", "offset": 2, "group_size": 4}],
 "collection": {"scores": false, "full_vocabulary_steps": []}}
EOF
```

Relative paths in `execution.json` resolve from its own directory.

## Prepare

```sh
invar use prepare --declaration declaration.json --execution execution.json --output contract.json
```

It prints `"status": "prepared"`, 8 inputs in 4 units with two seeds per unit, and 2 candidate executions, and it restates each decision under `owner_review`, for example "The candidate may raise the mean task loss by at most 1/8 over the reference, because losing more than one of the eight answers would make evaluation results depend on the numerics." With an empty `reliance` list it refuses instead, listing each of the eleven premises from `ParameterMeaning` to `SelectionControl` as `missing-reliance`.

```sh
invar use plan --contract contract.json --units 4
```

It answers `"standard": "finite_domain"` with no bound: a finite domain is decided by its exact mean, so there is no sample size to plan.

## Run

```sh
invar use run --contract contract.json --execution execution.json --output run
```

It checks the adapters against the two policies, then runs six finite batches, two per side and two for the repeat. Each batch loads the model once, and the whole run took 293 seconds. The output is `"status": "recorded"`, and `run` holds `contract.json`, `plan.json`, `records.json`, `runs.json` and every batch log under `logs`.

## Admit

```sh
cd run
invar use admit --contract contract.json --runs runs.json > decision.json
jq .decision.status decision.json
```

`decision.status` is `admitted_under_declared_reliance`. Both sides answered all eight questions, so the reference loss, the candidate loss and the increase are all 0, within the ceilings of 1/2 and 1/8. The candidate's behavior bits were identical in both schedules for all eight inputs, which meets the invariance requirement. The decision lists 117 remaining conditions, each a premise that rests on the owner's declared reliance, attached to the input, side or batch it covers.

The observation also shows how the two numerics differ, as diagnostics that do not enter the decision. The stock and row independent profiles produced the same tokens for seven of the eight inputs, and identical probability bits for none. The largest log ratio over a shared prefix was 0.25.

Running `invar use admit` again on the same directory prints the same bytes.
