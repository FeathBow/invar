import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import evaluation
from quality import compare

SEEDS = (17, 29, 43, 71)
TOKEN_LIMIT = 4
INITIAL_POLICY = "a" * 64
TRAINED_POLICY = "b" * 64


def declarations():
    return [{"tasks": [{"name": f"sample/{seed}", "group": "question", "prompt": "Compute one plus one.",
                        "seed": seed, "tokens": TOKEN_LIMIT, "temperature": 0.8, "answer": "#### 2"}
                       for seed in seeds], "order": list(range(len(seeds))), "delivery": list(range(len(seeds)))}
            for seeds in (SEEDS[:2], SEEDS)]


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


def write(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="invar-quality-"))
        self.tasks = self.directory / "tasks.json"
        self.tasks.write_text(json.dumps(declarations()))
        self.initial = evaluation.Run(log=self.directory / "initial.jsonl", policy=INITIAL_POLICY, exit_code=0)
        self.trained = evaluation.Run(log=self.directory / "trained.jsonl", policy=TRAINED_POLICY, exit_code=0)
        self.values = records(TRAINED_POLICY, ((1, 1), (0, 1, 0, 0)))
        write(self.initial.log, records(INITIAL_POLICY, ((0, 1), (0, 0, 0, 0))))
        write(self.trained.log, self.values)

    def test_weighted_pairs_preserve_cohort_local_names_and_seed_groups(self):
        result = compare(self.tasks, self.initial, self.trained)
        overall = result["overall"]
        self.assertEqual(overall["initial"]["reward_sum"], 1)
        self.assertEqual(overall["trained"]["reward_sum"], 3)
        self.assertEqual(overall["initial"]["reward_mean"], 1 / 6)
        self.assertEqual(overall["reward_mean_change"], 2 / 6)
        self.assertEqual((overall["improved_samples"], overall["worsened_samples"], overall["unchanged_samples"]), (2, 0, 4))
        self.assertEqual([group["comparison"]["initial"]["sample_count"] for group in result["by_group"]], [2, 4])
        self.assertEqual([group["comparison"]["reward_mean_change"] for group in result["by_group"]], [0.5, 0.25])
        self.assertEqual([row["seed"] for row in result["by_seed"]], list(SEEDS))
        self.assertEqual(result["by_seed"][1]["comparison"]["trained"]["reward_sum"], 2)
        self.assertEqual(result["zero_variance_groups"], {"initial": 1, "trained": 1})
        self.assertEqual(result["tasks_sha256"], hashlib.sha256(self.tasks.read_bytes()).hexdigest())
        self.assertEqual(result["trained"]["log_sha256"], hashlib.sha256(self.trained.log.read_bytes()).hexdigest())

    def test_report_delivery_order_does_not_change_sample_pairing(self):
        before = compare(self.tasks, self.initial, self.trained)
        self.values[0]["samples"].reverse()
        self.values[1]["samples"].reverse()
        write(self.trained.log, self.values)
        after = compare(self.tasks, self.initial, self.trained)
        for field in ("overall", "by_group", "by_seed"):
            self.assertEqual(before[field], after[field])

    def test_full_worker_stream_preserves_summaries_and_log_identity(self):
        before = compare(self.tasks, self.initial, self.trained)
        stages = ("loading", "profile", "load", "loaded_adapter", "consumed", "inference", "result")
        worker = [{"stage": stage} for stage in stages]
        initial = records(INITIAL_POLICY, ((0, 1), (0, 0, 0, 0)))
        write(self.initial.log, [*worker, initial[0], *worker, *initial[1:]])
        after = compare(self.tasks, self.initial, self.trained)
        for field in ("overall", "by_group", "by_seed"):
            self.assertEqual(before[field], after[field])
        self.assertEqual(after["initial"]["log_sha256"], hashlib.sha256(self.initial.log.read_bytes()).hexdigest())
        self.assertNotEqual(before["initial"]["log_sha256"], after["initial"]["log_sha256"])

    def test_declared_session_count_must_match_recorded_model_loads_of_every_cohort(self):
        stages = ("loading", "profile", "load", "loaded_adapter", "consumed", "inference", "result")
        worker = [{"stage": stage} for stage in stages]
        initial = records(INITIAL_POLICY, ((0, 1), (0, 0, 0, 0)))
        two = [*worker, *worker, initial[0], *worker, *worker, *initial[1:-1], {**initial[-1], "sessions": 2}]
        write(self.initial.log, two)
        declared = compare(self.tasks, self.initial, self.trained)
        write(self.initial.log, [*worker, initial[0], *worker, *initial[1:-1], {**initial[-1], "sessions": 1}])
        sequential = compare(self.tasks, self.initial, self.trained)
        write(self.initial.log, [*worker, initial[0], *worker, *initial[1:]])
        undeclared = compare(self.tasks, self.initial, self.trained)
        for field in ("overall", "by_group", "by_seed"):
            self.assertEqual(declared[field], undeclared[field])
            self.assertEqual(sequential[field], undeclared[field])
        for count in (1, 3, 0, -2, "2", 2.5, None):
            write(self.initial.log, [*two[:-1], {**initial[-1], "sessions": count}])
            with self.subTest(sessions=count), self.assertRaises(ValueError):
                compare(self.tasks, self.initial, self.trained)
        write(self.initial.log, [*worker, *worker, initial[0], *worker, *initial[1:-1], {**initial[-1], "sessions": 2}])
        with self.assertRaises(ValueError):
            compare(self.tasks, self.initial, self.trained)
        write(self.initial.log, [*worker, initial[0], *worker, *initial[1:-1], *worker, {**initial[-1], "sessions": 1}])
        with self.assertRaises(ValueError):
            compare(self.tasks, self.initial, self.trained)

    def test_unknown_worker_records_and_output_after_completion_fail(self):
        invalid = [{"stage": "unknown"}, {"diagnostic": "unclassified"}, [],
                   {"phase": "evaluation", "stage": "result"}]
        for row in invalid:
            write(self.trained.log, [row, *self.values])
            with self.subTest(record=row), self.assertRaises(ValueError):
                compare(self.tasks, self.initial, self.trained)
        write(self.trained.log, [*self.values, {"stage": "profile"}])
        with self.assertRaises(ValueError):
            compare(self.tasks, self.initial, self.trained)

    def test_worker_json_errors_are_not_hidden_by_summary_selection(self):
        for prefix in ('{"stage":"load","stage":"profile"}\n', '{"stage":\n'):
            write(self.trained.log, self.values)
            self.trained.log.write_text(prefix + self.trained.log.read_text())
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                compare(self.tasks, self.initial, self.trained)

    def test_truncation_and_zero_variance_groups_remain_in_denominators(self):
        self.values[1]["samples"][0]["truncated"] = True
        self.values[1]["summary"]["truncated_count"] = 1
        write(self.trained.log, self.values)
        result = compare(self.tasks, self.initial, self.trained)
        self.assertEqual(result["overall"]["trained"]["sample_count"], 6)
        self.assertEqual(result["overall"]["trained"]["truncation_rate"], 1 / 6)
        self.assertEqual(result["overall"]["trained"]["mean_response_tokens"], TOKEN_LIMIT)
        self.assertEqual(result["by_group"][1]["comparison"]["initial"]["reward_sum"], 0)

    def test_process_failure_cannot_be_hidden_by_a_completion_record(self):
        for status in (7, -9, True):
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "process"):
                compare(self.tasks, self.initial, replace(self.trained, exit_code=status))

    def test_missing_repeated_trailing_and_unterminated_records_fail(self):
        variants = [self.values[:-1], self.values + self.values[-1:], self.values[1:],
                    [self.values[1], self.values[0], self.values[2]]]
        for values in variants:
            write(self.trained.log, values)
            with self.subTest(records=len(values)), self.assertRaises(ValueError):
                compare(self.tasks, self.initial, self.trained)
        write(self.trained.log, self.values)
        self.trained.log.write_bytes(self.trained.log.read_bytes().rstrip(b"\n"))
        with self.assertRaisesRegex(ValueError, "final evaluation line"):
            compare(self.tasks, self.initial, self.trained)

    def test_sample_policy_seed_inventory_and_reused_bindings_fail(self):
        variants = []
        for field, value in (("seed", 18), ("reward", True), ("reward", float("nan")),
                             ("response_tokens", TOKEN_LIMIT + 1), ("truncated", True), ("name", "unknown")):
            candidate = copy.deepcopy(self.values)
            candidate[0]["samples"][0][field] = value
            variants.append(candidate)
        wrong_policy = copy.deepcopy(self.values)
        wrong_policy[1]["policy"] = INITIAL_POLICY
        repeated = copy.deepcopy(self.values)
        repeated[1]["samples"][0]["binding"] = repeated[0]["samples"][0]["binding"]
        variants.extend((wrong_policy, repeated))
        for candidate in variants:
            write(self.trained.log, candidate)
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                compare(self.tasks, self.initial, self.trained)

    def test_summary_cannot_disagree_with_complete_sample_records(self):
        for field in self.values[0]["summary"]:
            candidate = copy.deepcopy(self.values)
            candidate[0]["summary"][field] += 1
            write(self.trained.log, candidate)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "summary"):
                compare(self.tasks, self.initial, self.trained)

    def test_frozen_input_changes_and_duplicate_json_fields_are_rejected(self):
        changed = declarations()
        changed[0]["tasks"][0]["seed"] += 1
        self.tasks.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "input identity"):
            compare(self.tasks, self.initial, self.trained)
        self.tasks.write_text(json.dumps(declarations()))
        text = self.trained.log.read_text().replace('"phase":', '"phase": "ignored", "phase":', 1)
        self.trained.log.write_text(text)
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            compare(self.tasks, self.initial, self.trained)

    def test_input_digest_binds_prompt_answer_temperature_and_exact_bytes(self):
        for field, value in (("prompt", "A different question."), ("answer", "#### 7"), ("temperature", 0.7)):
            changed = declarations()
            changed[0]["tasks"][0][field] = value
            self.tasks.write_text(json.dumps(changed))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "input identity"):
                compare(self.tasks, self.initial, self.trained)
        self.tasks.write_text(json.dumps(declarations(), indent=2))
        with self.assertRaisesRegex(ValueError, "input identity"):
            compare(self.tasks, self.initial, self.trained)

    def test_cli_reports_valid_summary_and_rejects_a_failed_process(self):
        command = [sys.executable, "-B", str(Path(__file__).resolve().parents[1] / "quality.py"), "--tasks", str(self.tasks)]
        for name, run in (("initial", self.initial), ("trained", self.trained)):
            command += [f"--{name}-log", str(run.log), f"--{name}-policy", run.policy, f"--{name}-exit-code", "0"]
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["overall"]["reward_sum_change"], 2)
        command[-1] = "7"
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
