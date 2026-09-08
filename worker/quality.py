import argparse
import json
from pathlib import Path

import evaluation


def metrics(samples):
    count = len(samples)
    reward = sum(item.reward for item in samples)
    tokens = sum(item.response_tokens for item in samples)
    truncated = sum(item.truncated for item in samples)
    return {"sample_count": count, "reward_sum": reward, "response_tokens": tokens,
            "truncated_count": truncated, "reward_mean": reward / count,
            "mean_response_tokens": tokens / count, "truncation_rate": truncated / count}


def summarize(pairs):
    initial = metrics(tuple(first for first, _ in pairs))
    trained = metrics(tuple(second for _, second in pairs))
    change = trained["reward_sum"] - initial["reward_sum"]
    return {"initial": initial, "trained": trained,
            "reward_sum_change": change, "reward_mean_change": change / len(pairs),
            "improved_samples": sum(second.reward > first.reward for first, second in pairs),
            "worsened_samples": sum(second.reward < first.reward for first, second in pairs),
            "unchanged_samples": sum(second.reward == first.reward for first, second in pairs)}


def grouped(pairs, key):
    groups = {}
    for pair in pairs:
        groups.setdefault(key(pair[0]), []).append(pair)
    return {name: summarize(tuple(values)) for name, values in sorted(groups.items())}


def constant_groups(groups, side):
    return sum(value[side]["reward_sum"] in (0, value[side]["sample_count"]) for value in groups.values())


def compare(definition, initial, trained):
    tasks = evaluation.tasks(definition)
    first_digest, first = evaluation.read(initial, tasks)
    second_digest, second = evaluation.read(trained, tasks)
    pairs = tuple(zip(first, second, strict=True))
    seeds = grouped(pairs, lambda item: item.seed)
    groups = grouped(pairs, lambda item: (item.cohort, item.group))
    return {"comparison": "paired evaluation summaries", "tasks_sha256": tasks.digest,
            "initial": {"policy": initial.policy, "log_sha256": first_digest, "exit_code": initial.exit_code},
            "trained": {"policy": trained.policy, "log_sha256": second_digest, "exit_code": trained.exit_code},
            "overall": summarize(pairs),
            "group_count": len(groups),
            "zero_variance_groups": {side: constant_groups(groups, side) for side in ("initial", "trained")},
            "by_seed": [{"seed": seed, "comparison": result} for seed, result in seeds.items()],
            "by_group": [{"cohort": cohort, "group": group, "comparison": result}
                         for (cohort, group), result in groups.items()],
            "scope": "complete reported evaluations and supplied process exit status; not report authenticity, numerical qualification, or statistical generalization"}


def arguments():
    parser = argparse.ArgumentParser(description="Compare complete initial and trained-policy evaluation reports")
    parser.add_argument("--tasks", type=Path, required=True, help="Identical frozen input used by both evaluations")
    for side in ("initial", "trained"):
        parser.add_argument(f"--{side}-log", type=Path, required=True, help="Complete invar evaluate stdout")
        parser.add_argument(f"--{side}-policy", required=True, help="Expected canonical adapter identity")
        parser.add_argument(f"--{side}-exit-code", type=int, required=True, help="Independently observed evaluation process exit status")
    values = vars(parser.parse_args())
    runs = tuple(evaluation.Run(**{field: values[f"{side}_{field}"] for field in ("log", "policy", "exit_code")})
                 for side in ("initial", "trained"))
    return values["tasks"], *runs


def main():
    print(json.dumps(compare(*arguments()), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
