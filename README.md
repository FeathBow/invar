# Invar

Invar is developing an inference and reinforcement-learning runtime around a shared typed semantic core and explicit execution evidence. It aims to unify inference, rollout and learning under one dependency and evidence discipline.

The current implementation is a pure request-transition model with history checks and mutation counterexamples. Real-model inference and learning are not implemented yet. See the [request specification](docs/spec.md) for its rules, observations and evidence limits.

## Development

Build with GHC 9.14.1 and cabal-install 3.18.1.0:

```sh
cabal build all --enable-tests
cabal test all --test-show-details=direct
```

Source checks use Fourmolu 0.20.1.0 and HLint 3.10. Tests exercise Invar's request semantics; passing them is not a machine-checked proof or a GPU correctness guarantee.

```sh
fourmolu --mode check --record-brace-space true src test
hlint src test
```
