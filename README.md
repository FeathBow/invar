# Invar

Invar decides when a new numerical implementation may replace the reference in an LLM training loop, and records the evidence behind every decision.

## Why this is needed

A faster kernel is cheap to write today. An agent can produce dozens and check each one against sample inputs and outputs. The hard part begins when one of them runs inside a whole model. Optimized kernels rarely match the reference bit for bit, and a small difference in one operator can grow over hundreds of decoding steps until the model's output distribution shifts or a sampled token changes. Whether that change is acceptable depends on what the model is used for, so the check has to be made end to end, on the model's real task.

Invar turns that judgement into a contract. The contract names the prompts the model will see, how each response is scored, how much the score may drop and with what confidence, which results must stay exactly the same, and which outside assumptions the decision may rest on. Invar runs the reference and the candidate, checks every record they report, and gives one of three answers: the candidate is admitted, the contract is violated, or the evidence is too thin to decide.

## What it measures

For each prompt, Invar compares the two implementations token by token. It finds the first token where they diverge, checks whether their probabilities are identical down to the last bit, and adds up the log probability difference over the tokens they share. To see how the candidate behaves on the same context after the outputs split, it feeds the reference's own response through the candidate and scores every step. At chosen steps it captures the probability of every token in the vocabulary from both sides and bounds the KL divergence between them in each direction.

These measurements sit next to the task score. A candidate can show a large KL on a few tokens while its task score stays within bounds, and the contract states which of the two matters for the use at hand. The task bound is a confidence bound over the sampled prompts, so the answer covers the whole population of prompts at the stated confidence.

## A first result

The first acceptance ran Qwen3.8-27B on one GH200. The reference was vLLM with FP32 LoRA. Stock vLLM can give one prompt slightly different numbers depending on which other requests share its batch, because the order of floating point additions follows the batch layout. The candidate removed that dependence with two changes: NF4 matrix multiplication always takes the same kernel path, and softmax computes each row on its own in fixed blocks of 1024 columns.

The candidate changed the tokens of 649 of 2000 GSM8K problems drawn after the contract was fixed, and the contract still admitted it. The task loss rose by 0.0005, with an upper bound of 0.0202 at 97.5% confidence against a ceiling of 0.025. All 2000 problems gave identical bits across three runs that grouped the requests differently. Every remaining assumption is recorded with the party that vouches for it and a digest of its basis.

Running the loop through Invar costs at most 1.1 times running the same workers without it, and at most 1.1 times using vLLM or MLX directly. This held for inference and for full training cycles, on CUDA and on Apple Silicon.

## How it works

Invar is a reinforcement learning framework with a Haskell core that owns the training loop. In each cycle the core collects responses for a declared set of prompts, scores them, computes advantages and the training objective, runs one update and publishes a checkpoint that the next round of generation must load. The core computes rewards, advantages and the objective itself and compares them with what the workers report.

Python workers do the heavy computation: vLLM generates and Hugging Face trains on CUDA, and MLX does both on Apple Silicon. The core talks to every worker through one protocol, so another engine joins by implementing that protocol.

The command line covers training (`invar train`), evaluation of a fixed policy (`invar evaluate`), single inference calls (`invar infer`), the measurements above (`invar compare numerical` and `invar score`) and the admission decision (`invar use admit`). `invar compare histories` checks whether two training runs that scheduled their work differently published the same policies.

## Documentation

- [Contract](docs/design/contract.md): requests, programs, use contracts, evidence and admission.
- [Loop](docs/design/loop.md): the training loop.
- [vLLM backend](docs/backends/vllm.md): the CUDA worker with NF4 and FP32 LoRA.
- [MLX backend](docs/backends/mlx.md): the Apple Silicon worker.
- [Acceptance](docs/results/acceptance.md): the first acceptance and how its values were set.
- [Performance](docs/results/performance.md): the 1.1 bound and its results.
- [Patches](patches/README.md): the pinned vLLM and plugin patches.

## Development

Toolchain: GHC 9.14.1, cabal-install 3.18.1.0, Fourmolu 0.20.1.0 ([fourmolu.yaml](fourmolu.yaml)) and HLint 3.10. Worker dependencies are pinned in [worker/requirements.txt](worker/requirements.txt) and [worker/locks/](worker/locks). CI runs:

```sh
cabal build all --enable-tests
cabal test all --test-show-details=direct
cabal exec -- sh test/types.sh ghc
fourmolu --mode check $(git ls-files '*.hs')
hlint src test app
cabal exec -- ghc -XGHC2021 -Wall -Werror -package invar -outputdir dist-newstyle/confidence -o dist-newstyle/confidence/reference test/confidence/Main.hs
python3 test/confidence/check.py dist-newstyle/confidence/reference
export PATH="$(dirname "$(cabal list-bin exe:invar)"):$PATH"
python -B -m unittest worker.tests.invocation worker.tests.probepacked worker.tests.probestore
python -B -m unittest discover -s worker/tests/hf -t . -p '*.py'
```

`test/types.sh` compiles every fixture in `test/accept` and requires each fixture in `test/reject` to fail with the diagnostic in its `-- Reject:` line. `test/confidence/check.py` compares the core's Hoeffding and empirical Bernstein bounds with 120 digit decimal arithmetic and requires each to be sound and within 1e-12. The `worker/tests/mlx` and `worker/tests/vllm` packages run the same way on Apple Silicon and on a CUDA host.

Contributions follow [CONTRIBUTING.md](CONTRIBUTING.md) under [Apache-2.0](LICENSE).
