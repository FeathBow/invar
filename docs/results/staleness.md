# Staleness

On one GH200, a GRPO run whose rollouts came from the policy one update behind (staleness one) was compared with a run at staleness zero, from the same initial checkpoint and with the same settings. Under the reliance it declares, a use contract found that the staleness-one checkpoint meets the requirements for replacing the staleness-zero checkpoint on the 1255 declared GSM8K test problems with two seeds each, and the staleness-one run took 0.745 times the wall time of the staleness-zero run. The two runs form one pair in a fixed order, so the result describes these two checkpoints on this deployment. [The loop design](../design/loop.md) describes what the core fixes when rollouts and updates overlap.

## Fixed before the runs

Before training, the experiment lead confirmed a frozen record of the use, both ceilings and their reasons, the problems, the training budget, the checkpoint selection rule, and the failure and exclusivity rules. The use contract was frozen after training and before any evaluation.

| Setting | Value |
| --- | --- |
| model | `Qwen/Qwen3.5-4B` with a LoRA adapter, resident vLLM inference and a resident Hugging Face learner, as separate processes on one GPU |
| training | GRPO from one initial checkpoint: 16 updates of 4 GSM8K training problems with 4 samples each, 1024 response tokens, temperature 0.8, clip 0.2, KL penalty 0.04, AdamW with learning rate 0.0001, one optimizer step per update, reference probabilities from the engine |
| runs | staleness zero, then staleness one, in one allocation on one GPU, differing only in `--staleness` |
| checkpoint | the last published generation of each run |
| evaluation | the 1255 GSM8K test problems left after excluding 64 that an earlier experiment evaluated, each with 2 paired seeds and 1024 response tokens; one unit per problem; exact decimal scoring, with a truncated response counted as wrong |
| contract | finite domain; reference loss at most 1/5, because the lowest usable accuracy is 80%; mean loss increase at most 1/40, so the staleness-one checkpoint may answer at most 2.5 fewer of every 100 problems |

The training problems are disjoint from the evaluation problems and from every problem used to prepare the runs, by row and question digest. A watcher listed the device's compute processes every two seconds during training and evaluation and found only the experiment's own processes.

## Quality

| Checkpoint | Mean loss | Accuracy |
| --- | --- | --- |
| staleness zero (reference) | 55/502 | 89.0% |
| staleness one (candidate) | 26/251 | 89.6% |

The mean loss increase is −3/502, about −0.006, and the reference loss is within its ceiling, so the contract returned `admitted_under_declared_reliance`. The decision covers these 1255 problems and their declared seeds, and holds when the declared reliance holds. The staleness-one checkpoint answered about 0.6 percentage points more of them correctly; that difference is an observation on this finite domain and describes these problems and seeds only. Its remaining conditions record each declared premise with the party that vouches for it, and the authenticity of every execution report on both sides. The population premises stay unestablished, so the result carries no confidence bound for other problems or seeds.

## Throughput and resources

| Run | Wall time | Response tokens consumed | Tokens per second |
| --- | --- | --- | --- |
| staleness zero | 3408 s | 86823 | 25.5 |
| staleness one | 2539 s | 86947 | 34.2 |

Each run consumed 256 samples. Wall time covers the whole training command: loading, rollout, reference scoring, waiting, retries and publication. The staleness-one run's samples differ from the staleness-zero run's from the second cohort on, because each cohort comes from an earlier policy. With one pair in a fixed order, an order effect cannot be told apart from the effect of staleness.

Training took one GPU for 1 hour 39 minutes, with an observed peak of 42032 MiB of the 97871 MiB on the device, sampled every 500 ms. The evaluation took one GPU for 5 hours 18 minutes for 2510 generations on each side, with an observed peak of 6722 MiB. The admission read the retained logs on CPU in 4 minutes 15 seconds with 2.9 GiB of memory.

## Overlap

A separate three-update run at staleness one was recorded with a CUDA trace, and every kernel was attributed to the call that launched it. Kernels of the next cohort's generation started between kernels of the running update, so generation advanced while the update ran. The two processes time-shared the device, and tracing slowed the workers, so that run is left out of the timings above.

## Revision of the admission

The frozen admission ran out of memory. Its core read each run's whole batch log and replayed the whole batch for every member, which grew with the number of runs times the size of the batch. The admission was repeated with a revised core that reads, replays and indexes each batch log once. Before that, the frozen and the revised core admitted a complete earlier evaluation of 256 paired runs with identical output, including every observation and the decision. A record written after the evaluation binds the original freezes and confirmation, the revised source and binary, and the admission inputs. The training and evaluation records are the original ones; the admission departs from the frozen procedure in its program, and the original failure rule allows no rerun with a changed configuration within the round, so this result reports that departure.
