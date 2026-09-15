import hashlib
import json

SEEDS = (17, 29, 43, 71)
TOKEN_LIMIT = 4
INITIAL_POLICY = "a" * 64
TRAINED_POLICY = "b" * 64


def declarations():
    return [{"tasks": [{"name": f"sample/{seed}", "group": "question", "prompt": "Compute one plus one.",
                        "seed": seed, "tokens": TOKEN_LIMIT, "temperature": 0.8, "answer": "#### 2"}
                       for seed in seeds], "order": list(range(len(seeds))), "delivery": list(range(len(seeds)))}
            for seeds in (SEEDS[:2], SEEDS)]


def write(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def records(policy, rewards):
    result, ordinal = [], 0
    for index, (tasks, scores) in enumerate(zip(declarations(), rewards, strict=True)):
        samples = [{"name": task["name"], "group": task["group"], "seed": task["seed"], "reward": reward,
                    "response_tokens": TOKEN_LIMIT, "truncated": False,
                    "binding": dict.fromkeys(("call", "attempt", "instance"), ordinal + position)}
                   for position, (task, reward) in enumerate(zip(tasks["tasks"], scores, strict=True))]
        count = len(scores)
        summary = {"sample_count": count, "reward_sum": sum(scores), "response_tokens": count * TOKEN_LIMIT,
                   "truncated_count": 0, "group_count": 1, "zero_variance_groups": int(len(set(scores)) == 1)}
        result.append({"phase": "evaluation", "cohort": index, "policy": policy, "summary": summary, "samples": samples})
        ordinal += count
    digest = hashlib.sha256(json.dumps(declarations()).encode()).hexdigest()
    return [*result, {"phase": "evaluation_complete", "policy": policy, "cohorts": len(result), "tasks_sha256": digest}]
