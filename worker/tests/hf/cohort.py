import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import unittest

from worker.cohort import DISTILLATION, SPECIFICATION, Distillation, Distilled, decode
from worker.hf.step import optimizer_options


def request():
    samples = [{"sample": "b", "group": "question", "prompt": "Compute the answer.",
                "seed": 17, "limit": 1, "temperature": 0.8,
                "tokens": [11, 12], "prompt_length": 1, "version": 0, "behavior_policy": "a" * 64, "behavior_bits": [0xBF800000], "reference_bits": [0xBFA00000],
                "text": "#### 437", "truncated": False, "reward": 1.0, "advantage_bits": 0x3F7FF2E5},
               {"sample": "a", "group": "question", "prompt": "Compute the answer.",
                "seed": 18, "limit": 1, "temperature": 0.8,
                "tokens": [11, 13], "prompt_length": 1, "version": 0, "behavior_policy": "a" * 64, "behavior_bits": [0xBF000000], "reference_bits": [0xBF400000],
                "text": "#### 438", "truncated": False, "reward": 0.0, "advantage_bits": 0xBF7FF2E5}]
    return {"specification": SPECIFICATION, "policy": "a" * 64, "learner": "b" * 64,
            "reference": "c" * 64, "reference_source": "engine", "tokenizer": "d" * 64, "base": "e" * 64, "assembly": "f" * 64,
            "behavior_model": {"base": "0" * 64, "assembly": "1" * 64},
            "schedule": {"update": 0, "staleness": 0}, "samples": samples, "order": ["a", "b"], "steps": [["a", "b"]],
            "epsilon": 0.2, "penalty": 0.04, "delta": 1e-4,
            "optimizer": {"learning_rate": 1e-4, "betas": [0.9, 0.999],
                          "epsilon": 1e-8, "weight_decay": 0.0}}


def distilled():
    value = request()
    del value["delta"]
    teacher = {"model": "teacher-model", "revision": "teacher-revision", "adapter": "6" * 64, "tokenizer": value["tokenizer"], "base": "7" * 64,
               "assembly": "8" * 64, "temperature": 0.8, "truncation": "none", "logprobs": "processed", "path": "forced"}
    samples = [{**{key: item for key, item in sample.items() if key not in ("group", "reward", "advantage_bits")},
                "teacher_bits": [0xBF400000], "teacher_call": {"call": 5, "attempt": 5, "instance": index}}
               for index, sample in enumerate(value["samples"])]
    return {**value, "specification": DISTILLATION, "teacher": teacher, "samples": samples}


def synchronous(value):
    return {**value, "samples": [{**item, "version": value["schedule"]["update"], "behavior_policy": value["policy"]}
                                 for item in value["samples"]]}


class CohortTests(unittest.TestCase):
    def test_declared_order_is_distinct_from_delivery_sequence(self):
        parsed = decode(request())
        self.assertEqual(tuple(item.sample for item in parsed.samples), ("b", "a"))
        self.assertEqual(parsed.order, ("a", "b"))
        self.assertEqual(tuple(item.seed for item in parsed.samples), (17, 18))
        self.assertEqual(parsed.optimizer.betas, (0.9, 0.999))

    def test_payload_changes_after_decoding_do_not_change_request(self):
        value = request()
        parsed = decode(value)
        value["samples"][0]["prompt"] = "Changed"
        value["order"].reverse()
        value["optimizer"]["betas"][0] = 0.1
        value["samples"][0]["tokens"].append(14)
        value["samples"][0]["behavior_bits"][0] = 0
        value["behavior_model"]["base"] = "2" * 64
        self.assertEqual(parsed.samples[0].prompt, "Compute the answer.")
        self.assertEqual(parsed.order, ("a", "b"))
        self.assertEqual(parsed.optimizer.betas, (0.9, 0.999))
        self.assertEqual(parsed.samples[0].tokens, (11, 12))
        self.assertEqual(parsed.samples[0].behavior_bits, (0xBF800000,))
        self.assertEqual(parsed.behavior_model.base, "0" * 64)

    def test_behavior_model_is_explicit_and_distinct_from_the_learner(self):
        parsed = decode(request())
        self.assertNotEqual(parsed.behavior_model.base, parsed.base)
        self.assertNotEqual(parsed.behavior_model.assembly, parsed.assembly)
        invalid = [None, {}, {"base": "0" * 64}, {"assembly": "1" * 64},
                   {"base": "0" * 64, "assembly": "1" * 64, "extra": 0},
                   {"base": True, "assembly": "1" * 64}, {"base": "0" * 64, "assembly": "G" * 64}]
        for model in invalid:
            with self.subTest(model=model), self.assertRaises(ValueError):
                decode({**request(), "behavior_model": model})

    def test_observed_tokens_and_probability_bits_are_validated(self):
        changes = [("tokens", [11]), ("tokens", [11, -1]), ("tokens", [11, True]),
                   ("prompt_length", 0), ("prompt_length", 2), ("behavior_bits", []),
                   ("behavior_bits", [0x7FC00000]), ("behavior_bits", [0x3F800000]),
                   ("behavior_bits", [1 << 32]), ("behavior_bits", [False]),
                   ("truncated", 1), ("reward", float("inf")),
                   ("advantage_bits", True), ("advantage_bits", -1), ("advantage_bits", 1 << 32),
                   ("advantage_bits", 0x7FC00000), ("advantage_bits", 0x7F800000)]
        for field, value in changes:
            changed = request()
            changed["samples"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                decode(changed)
        changed = request()
        changed["samples"][0]["behavior_bits"] = [0x80000000]
        word = decode(changed).samples[0].behavior_bits[0]
        self.assertEqual(word, 0x80000000)

    def test_samples_follow_the_schedule_and_their_behavior_policy(self):
        base = request()
        earlier = {**base, "policy": "9" * 64, "schedule": {"update": 1, "staleness": 1},
                   "samples": [{**item, "behavior_policy": "c" * 64, "reference_bits": []} for item in base["samples"]]}
        decoded = decode(earlier)
        self.assertEqual((decoded.schedule.update, decoded.schedule.staleness, decoded.schedule.version()), (1, 1, 0))
        self.assertEqual({item.behavior_policy for item in decoded.samples}, {"c" * 64})
        decode({**earlier, "samples": [{**item, "behavior_policy": "e" * 64, "reference_bits": [0xBFA00000]} for item in base["samples"]]})
        startup = decode({**base, "schedule": {"update": 0, "staleness": 1}})
        self.assertEqual({item.version for item in startup.samples}, {0})
        later = decode({**earlier, "schedule": {"update": 3, "staleness": 1},
                        "samples": [{**item, "version": 2} for item in earlier["samples"]]})
        self.assertEqual((later.schedule.version(), {item.version for item in later.samples}), (2, {2}))
        invalid = [{**earlier, "samples": [{**item, "reference_bits": [0xBFA00000]} for item in earlier["samples"]]},
                   {**earlier, "samples": [{**item, "behavior_policy": "e" * 64} for item in earlier["samples"]]},
                   {**earlier, "schedule": {"update": 2, "staleness": 1}},
                   {**earlier, "samples": [{**item, "version": 1} for item in earlier["samples"]]},
                   {**base, "samples": [{**item, "behavior_policy": "9" * 64} for item in base["samples"]]},
                   {**base, "schedule": {"update": 0, "staleness": True}},
                   {**base, "schedule": {"update": 0, "staleness": 1}, "samples": [{**item, "behavior_policy": "9" * 64} for item in base["samples"]]},
                   {**earlier, "samples": [{**earlier["samples"][0], "behavior_policy": "e" * 64, "reference_bits": [0xBFA00000]}, earlier["samples"][1]]},
                   {**base, "schedule": {"update": 0}}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode(value)

    def test_missing_extra_duplicate_and_incomplete_inputs_fail(self):
        variants = []
        for field in request():
            missing = request()
            del missing[field]
            variants.append(missing)
        variants += [{**request(), "extra": 1}, {**request(), "specification": "other"},
                     {**request(), "order": ["a", "a"]}, {**request(), "order": ["a"]},
                     {**request(), "order": ["b", "unknown"]}, {**request(), "samples": []},
                     {**request(), "steps": []}, {**request(), "steps": [["a"]]}, {**request(), "steps": [["a"], []]},
                     {**request(), "steps": [["a", "b", "unknown"]]}, {**request(), "steps": ["ab"]},
                     {**request(), "steps": [["a", 1]]}]
        duplicate = request()
        duplicate["samples"][1]["sample"] = "b"
        variants.append(duplicate)
        singleton = request()
        singleton["samples"][1]["group"] = "other"
        variants.append(singleton)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode(value)

    def test_invalid_numeric_and_identity_inputs_fail(self):
        changes = [("epsilon", value) for value in (True, 0, 1, float("nan"), float("inf"))]
        changes += [("penalty", -1), ("delta", 0), ("policy", "A" * 64),
                    ("learner", "b" * 63), ("reference", 1), ("tokenizer", "D" * 64),
                    ("base", "E" * 64), ("assembly", "f" * 63)]
        for field, value in changes:
            changed = {**request(), field: value}
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                decode(changed)
        for field, value in [("limit", 0), ("limit", True), ("seed", 1.5),
                             ("temperature", 0), ("sample", ""), ("prompt", None)]:
            changed = request()
            changed["samples"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                decode(changed)

    def test_a_distillation_request_carries_the_teacher_and_its_words_without_rewards_or_groups(self):
        parsed = decode(distilled())
        self.assertIsInstance(parsed, Distillation)
        self.assertTrue(all(isinstance(item, Distilled) for item in parsed.samples))
        self.assertEqual([item.teacher_bits for item in parsed.samples], [(0xBF400000,), (0xBF400000,)])
        self.assertEqual([item.teacher_call.instance for item in parsed.samples], [0, 1])
        self.assertEqual((parsed.teacher.model, parsed.teacher.path), ("teacher-model", "forced"))

    def test_a_distillation_request_refuses_a_teacher_or_teacher_words_outside_the_declared_rules(self):
        def teacher(**changes):
            value = distilled()
            return {**value, "teacher": {**value["teacher"], **changes}}

        def scored(**changes):
            value = distilled()
            return {**value, "samples": [{**value["samples"][0], **changes}, value["samples"][1]]}

        decode(teacher(tokenizer="e" * 64))
        cases = [(teacher(tokenizer="E" * 64), "SHA-256"), (teacher(model=""), "model and revision"),
                 (teacher(temperature=0), "temperature must be positive"), (teacher(truncation="top-k"), "forced path"),
                 (scored(teacher_bits=[0x3F000000]), "nonpositive"), (scored(teacher_bits=[0xBF400000, 0xBF400000]), "cover every response token"),
                 (scored(teacher_call={"call": 5, "attempt": 5, "instance": -1}), "nonnegative integers"),
                 ({**distilled(), "epsilon": 1}, "Invalid distillation coefficient configuration"),
                 ({**distilled(), "delta": 1e-4}, "Expected exactly these request fields"),
                 (scored(reward=0.0), "Expected exactly these request fields"),
                 (scored(group="question"), "Expected exactly these request fields")]
        for value, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                decode(value)

    def test_optimizer_configuration_is_not_replaced_by_defaults(self):
        custom = request()
        custom["optimizer"] = {"learning_rate": 0.002, "betas": [0.8, 0.95],
                               "epsilon": 1e-7, "weight_decay": 0.01}
        self.assertEqual(optimizer_options(decode(custom).optimizer),
                         {"lr": 0.002, "betas": (0.8, 0.95), "eps": 1e-7, "weight_decay": 0.01,
                          "foreach": False, "fused": False, "amsgrad": False, "maximize": False,
                          "capturable": False, "differentiable": False})
        for field, value in [("learning_rate", -1), ("betas", [0.9]),
                             ("betas", [True, 0.999]), ("betas", [0.9, 1]),
                             ("epsilon", 0), ("weight_decay", -1), ("fused", True)]:
            changed = request()
            changed["optimizer"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                decode(changed)


if __name__ == "__main__":
    unittest.main()
