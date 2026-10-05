# Contracts and admission

Invar decides whether a candidate implementation may replace a reference implementation for a declared use. The Haskell core owns every meaning in that decision: what a request computes, which inputs a program may read, which worker report belongs to which call, what a numerical comparison observed, and what evidence a use requires. Python workers run the models. The core judges their reports against a use contract and returns a decision with every external premise it rests on. The training loop that produces the same kinds of reports is described in [loop.md](loop.md).

## Requests

`Invar.Spec.Request` models generation as a transition system with guarded scheduling events such as `Admit`, `Decode`, `Preempt` and `Complete`. The decoder reads only the request's own prompt and tokens, so admission order, preemption and other requests never reach it, and any two histories that complete a request produce the tokens computed by `unroll`. A use contract states the same property for real engines as batch invariance.

## Semantic programs

The inference call, the reward rule, the learning update and a contract's measurement are all checked semantic programs: first order and pure, over Booleans, exact rationals, binary32 bit patterns, tokens, records, sequences and maps. A schema declares sources, primitives and sinks. `Semantic` sources are inputs the result may depend on, `Operational` sources describe execution choices, and `LogicalRandom` sources carry declared randomness such as a sampling seed. Each sink names a specification, a payload type and the sources its payload may depend on.

`Invar.Spec.Dependency` computes the sources each expression depends on and rejects any emission whose dependencies fall outside its sink's allowed set. Programs travel as text: `Invar.Spec.Artifact.load` parses, type checks and analyses the bytes and only then returns an abstract `Checked` handle, which keeps the bytes so later reports can be compared with the exact program that was checked. Inference, for example, emits to `infer` / `categorical-inference/v1` from `request`, `policy` and the random `sample`.

## Invocation binding and results

`Invar.Spec.Invocation` ties each report to one execution. A binding has a `CallId` for one logical emission, an `AttemptId` for one dispatch of it, and an `Instance` for the executor load that runs it. A runtime fixes one checked program and command position. `prepare` evaluates the program and stores the emission under a fresh call, `issue` binds an attempt and instance, `consume` accepts the worker's report only when binding, program bytes and payload all match, and `finish` retains the output.

A worker writes JSON lines. `Invar.Infer.Session` is a pure state machine that takes each line of a single, serial, batched or resident exchange, checks the order of stages and the exact fields of each record, advances the runtime at `consumed` and `result`, and decides which request is sent next. Only a complete, matching exchange that ends in a clean exit or an acknowledged resident release produces `Invar.Infer.Trajectory.Trajectory` values. `Trajectory` is opaque and exports no constructor; callers read the binding, the request, the declared constraint, the observed model and revision, the tokens, behavior words (the FP32 log probability of each sampled token, as raw bits), the text and the truncation flag through accessors. Its evidence is a canonical byte encoding of these values, and the runtime journals the SHA-256 of that encoding for each stored call. `Invar.Infer.Observation.admit` rebuilds the same result from a retained log.

## Numerical observations

`Invar.Numerical.observe` compares a reference run and a candidate run that consumed the same tokenizer, prompt, token limit, temperature and seed. It records the first divergent step, the log probability ratio over the shared prefix, both lengths and truncation flags, and whether tokens and behavior words are equal. Scores (`invar score`) and full vocabulary probes add path ratios and KL bounds.

A relation names what must hold between the sides: `tokens`, `behavior-bits`, `termination`, a bounded `prefix-log-ratio` or `path-log-ratio`, a bounded scored path ratio for either side, or `full-vocabulary-kl` in a chosen direction.

## Evidence

`Invar.Spec.Evidence` checks claims through an explicit graph. Each node pairs a claim (an external obligation, a numerical claim, a task loss claim, a conjunction or an implication) with a rule such as `Observe`, `ObserveTaskLoss` or `Conjoin`. `check` returns one of three verdicts. `Accept` carries a certificate with the conclusion, the claims it still assumes and the methods used. `Refute` carries the refuted claim and a witness. `Unknown` carries a problem such as `MissingEvidence`, too few units or an unresolved KL interval, and any Unknown child makes its parent Unknown.

An external obligation records something the core cannot check from the bytes it holds, such as whether a worker's report describes what the hardware ran. It names a predicate, the digest of the checked scope and the concrete source it concerns. Each numerical observation lists its own obligations, such as `execution-report-authenticity` for each side, and these stay in the certificate until a use contract names who vouches for them.

## Use contracts

A use contract is a JSON document with format `invar-use-contract/v2`. It has seven parts.

- The domain declares the input population before anything runs: for each input, its cohort and key, its aggregation unit, the prompt, token limit, temperature, seed and named parameters, plus the provenance of the inputs.
- The measurement is a semantic program whose sources are bound to observed result fields (such as `Response` or `Truncated`) or to input parameters. It emits one number in a declared range, which the orientation maps onto a loss in `[0, 1]`.
- The reference and candidate are policy descriptions naming model, revision, adapter, tokenizer, frozen base and model assembly.
- The criterion holds numerical requirements (relations between reference and candidate), a loss requirement (ceilings on the mean reference loss and the mean loss increase under a standard of evidence), and invariance requirements (relations between repeated candidate executions of one input, with the number of executions).
- The protocols describe how the contract was frozen, acceptance data isolated and selection controlled.
- The reliance list names, for each premise kind, an authority and the SHA-256 digest of its basis.
- The transfers list declares, for the reference or the candidate side, a previous implementation whose recorded evidence may stand for the one the contract names. The two descriptions may differ only in the model assembly, and each transfer states the relation it preserves, for example that the changed files do not alter numerical execution. Contracts in the original format `invar-use-contract` have no transfers.

Budgets and alphas are exact rationals. The standard of evidence is `finite_domain` (the exact mean over the declared units), `population_hoeffding` or `population_bernstein_mp2009` (an upper confidence bound for a named population), or `conditional_derivation`, which always yields Unknown.

## Observation for a use

`Invar.Use.Observation.observe` expects, for every declared input, a reference and candidate pair plus any repeated candidate executions. It checks that each run consumed its declared input and that each side used one policy description with a fresh binding for every execution. It then observes each pair numerically, compares each adjacent pair of candidate executions for invariance, and measures both sides.

A unit's loss is the mean loss of its members, and each metric is the mean over units, so every unit weighs the same and repeats add no samples. The metrics are the reference loss, the candidate loss and their difference.

For a population standard, the reference loss is bounded at `reference_alpha` and the loss increase at `regression_alpha`, whose sum must be at most `family_alpha`. With `n` units, range `R` (1 for a loss, 2 for an increase) and alpha `a`, the empirical Bernstein width of Maurer and Pontil (2009, Theorem 4) is the square root of `2 V ln(2/a) / n` plus `7 R ln(2/a) / (3(n - 1))`, where `V` is the sample variance. The upper bound is the mean plus the width, capped at 1, and all arithmetic uses exact rational upper bounds. A bound above its ceiling gives Unknown: the current bound does not establish the requirement.

## Admission

`invar use admit` turns the observation into a decision.

1. `Invar.Use.Finding.establish` builds one evidence graph whose root is the conjunction of every requirement: the two loss claims, one numerical claim for each input and requirement, and one invariance claim for each repeated pair.
2. `Evidence.check` evaluates the graph.
3. `Invar.Use.Admission.admit` validates the contract, matches it against the observation, and adds the contract's own premises, including `ContractFrozen`, `AcceptanceIsolation`, one `ScheduleVariation` premise for each repeated execution, which states that the repeats ran under different schedules, and one `ImplementationPreservation` premise for each transfer.
4. Every remaining assumption must be an external obligation whose premise kind the contract's reliance list names.

A transfer does not rewrite evidence. The finding still describes the implementation that ran, admission accepts it only on the declared side, and the preservation premise needs reliance like any other. The premise is bound to both implementations and to the contract's requirements, domain and measurement, so it states that every transferred requirement keeps holding over that scope. The core checks that the premise is bound and kept; the named authority answers for its truth. A relation shown on a narrower basis, such as equal tokens on a probe, cannot support the whole use. A counterexample found through a transfer belongs to the previous implementation, so the decision is unknown with the reason `TransferredViolation` rather than a violation of the named one; the finding keeps the witness. When a change leaves the model assembly unchanged, no transfer is needed, because the recorded description already equals the named one.

The decision status is `admitted_under_declared_reliance`, `observed_violation` or `unknown`. An admission lists the methods used and every remaining condition with the authority and basis digest the contract declared for it. A violation carries its witness, and an unknown decision lists reasons such as `MissingReliance`.

Both commands read the contract and the retained runs:

```sh
invar use inspect --contract contract.json --runs runs.json
invar use admit --contract contract.json --runs runs.json
```

`runs.json` has one entry per input: `[cohort, key, pair, repeats]`, where the pair holds `invar compare numerical` arguments and each repeat describes one further candidate execution. A run recorded in a batch log names the batch's calls file with `--calls`, and admission replays the whole batch under that declaration before it takes the run's member by binding. Runs that name the same log must name the same calls file and exit status.

The certificate records each confidence bound it used as part of its method, so `invar use admit` reports the reference and increase bounds from the decision itself. Beside them it puts other common tests on the same paired losses, at the contract's regression alpha and ceiling, without changing the decision: `Invar.Use.Statistics` gives the population bound the contract did not use, a paired Wald interval with the equivalence and noninferiority verdicts it implies against the ceiling, and, when every loss is zero or one, the exact one sided McNemar tail for the units that got worse. The Wald interval uses an upper enclosure of the normal quantile computed from an alternating series with a bounded remainder and an upper rational bound on pi.

## The exact decimal adapter

`Invar.Use.Decimal` is one measurement adapter. It declares one input per workload task with the prompt as its unit and the expected answer as the parameter `expected`, and uses the reward program as its measurement, so loss is one minus the exact decimal reward and a truncated response has loss one. The generic boundary needs no standard answer. Parameters have no built-in meaning and the measurement is any checked program over observed fields, so a contract can measure an event such as truncation, or carry numerical requirements alone with no measurement.

## The release contract

The release contract admits a candidate for Qwen3.8-27B rollouts and evaluation on one GH200. The reference is vLLM with the FP32 LoRA patches. The candidate adds two changes to it so that each request's numbers do not depend on which requests share its batch: NF4 matrix multiplication always takes the same kernel path, and softmax reduces each row on its own in fixed blocks of 1024 columns. The contract uses the exact decimal measurement, the `population_bernstein_mp2009` standard with a family alpha of 1/20 split evenly, ceilings of 61/200 on the reference loss and 1/40 on the loss increase, and a `behavior-bits` invariance requirement over three candidate executions under different batch partitions. The measured values and the decision are in [acceptance.md](../results/acceptance.md), and the backends are described in [vllm.md](../backends/vllm.md) and [mlx.md](../backends/mlx.md).
