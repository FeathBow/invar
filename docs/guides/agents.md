# Operating Invar for a use owner

This guide is for whoever runs Invar on behalf of the person who owns a use decision, whether that operator is a person or an agent. The tool cannot tell who wrote a file, so every rule here applies to both.

## Who decides

The use owner decides the purpose, the inputs, the scoring method, the quality budgets and why they were chosen, the hard requirements, the protocols, and for every premise the authority who vouches for it and the files that support it. All of these sit under `decisions` in the declaration, and none of them has a default.

The operator collects artifact paths, writes the declaration from decisions the owner has stated, runs the commands, reads their JSON and explains it to the owner. When a decision is missing, the operator stops and asks the owner for it.

An operator never writes a budget, rationale, alpha, authority or basis the owner did not state. It never states a premise it cannot support, and never changes a seed, drops an input, relaxes a budget or reruns a command to get a better result.

## Isolation

Acceptance data must not influence how the candidate was generated or selected, or what the contract says. This holds before the contract is frozen and after it: choosing another candidate, editing the contract or filtering runs because of acceptance results all break isolation. Reading the inputs to carry out the measurement does not. An operator that did any of these things cannot supply the isolation premise, and says so to the owner.

## Commands

| Command | Success `status` | It means | It does not mean |
| --- | --- | --- | --- |
| `invar use prepare` | `prepared` | the contract is fully built and every required decision and reliance is present | that any premise is true, that the candidate is good, or that a run will succeed |
| `invar use plan` | `estimated` | a conditional estimate was computed from the stated assumptions | evidence of any kind |
| `invar use run` | `recorded` | every record the contract needs under this execution plan was produced and kept | that the candidate is admitted |
| `invar use admit` | read `decision.status` | the decision was computed | anything until `decision.status` is read |

`decision.status` is `admitted_under_declared_reliance`, `observed_violation` or `unknown`. Report it with the reasons and remaining premises that come with it.

`prepare`, `plan` and `run` print one JSON document:

```json
{"format": "invar-use-prepared-v1", "status": "refused", "means": "…", "does_not_mean": "…", "evidence": false,
 "problems": [{"code": "missing-decision", "at": "declaration:decisions.quality.regression_ceiling.rationale", "message": "…"}]}
```

`at` points to a command line argument (`argv:--contract`), a place in the declaration or execution file (`declaration:decisions.reliance[2]`), or a file the tool read (`artifact:path`). `evidence` is always `false` for these three commands.

## Problems and what to do

| Code | Exit | What to do |
| --- | --- | --- |
| `missing-argument` | 2 | add the argument |
| `missing-decision` | 2 | ask the owner for the decision at `at` |
| `invalid-value` | 2 | a value is outside its domain; fix a typo, or ask the owner when the value is theirs |
| `missing-reliance` | 2 | ask the owner who vouches for the premise and on what basis; never write a basis to get past this |
| `unsupported` | 2 | the method, standard, observation or backend is not available; report it to the owner |
| `schedule-unchanged` | 2 | give the repeat a different order or group size |
| `artifact-missing` | 3 | find the file or produce it with the documented command |
| `artifact-role` | 3 | use the file of the named role, such as `policy.json` in place of a learner checkpoint |
| `artifact-invalid` | 3 | the file does not parse or does not match its recorded identity; find the right file |
| `identity-mismatch` | 4 | a worker loaded something other than the contract's policy; report it and keep the records |
| `execution-failed` | 4 | report the failing group and its log; the records produced so far are kept |
| `internal-error` | 1 | report a defect in Invar; files the command wrote before it stopped may be incomplete, so do not use them |

After an exit 4 the operator reports the failure and does not rerun on its own. A rerun writes a new output directory and is the owner's decision.

## Example flow

```sh
invar use prepare --declaration declaration.json --execution execution.json --output contract.json
invar use plan --contract contract.json --units 64 --assumed-increase 0 --assumed-reference-loss 1/5
invar use run --contract contract.json --execution execution.json --output run
cd run
invar use admit --contract contract.json --runs runs.json
```

When `prepare` refuses with `missing-decision` or `missing-reliance`, the operator lists each `at` to the owner, waits for the answers, edits the declaration and runs `prepare` again. `plan` is optional and reads only the contract and the assumptions given to it. `run` writes into a directory that must not exist yet, and `admit` reads from inside it. The [walkthrough](walkthrough.md) runs this flow on Apple Silicon.
