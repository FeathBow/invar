# Walkthrough record

The workload, both runtime configurations and this declaration were written before any acceptance run.
Both sides share one adapter, initialized with seed 17; they differ only in the runtime configuration's numerics.
The candidate was not chosen, changed or retried on the basis of any acceptance result.
Scoring is exact-decimal from the Invar core; the expected answers are the ones in workload.json.
Every model run uses the pinned mlx-community/Qwen3.8-27B-4bit revision from the local cache and its own KV cache.
