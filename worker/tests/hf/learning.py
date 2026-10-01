import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import io
import json
import math
import unittest
from contextlib import redirect_stdout
from dataclasses import replace

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

from worker.tests import scalar
from worker.exchange import Exchange, observation
from worker.hf.learning import parameters, update
from worker.logical import Learner, Plan
from worker.hf.checkpoint import ADAM_BETAS, ADAM_EPSILON, LEARNING_RATE
from worker.hf.metrics import report
from worker.hf.model import adapter_state
from worker.hf.tensors import assert_equal, digest
from worker.hf.probability import words
from worker.trajectory import Request, Trajectory
from worker.tests.responder import Core, receiver

SMALL_ADVANTAGE = 2 ** -28
OVERFLOW_ADVANTAGE = 1e30
LOGICAL_ORDER = ("a", "b", "c")
ADVANTAGES = (1, -1, SMALL_ADVANTAGE)
BINDING = {"call": 7, "attempt": 11, "instance": 13}


def make_learner():
    base = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=False))
    model = get_peft_model(base, LoraConfig(r=1, lora_alpha=1, target_modules=["0"]))
    with torch.no_grad():
        for name, value in model.named_parameters():
            value.fill_(1 if "lora_A" in name else 0)
    optimizer = torch.optim.AdamW(parameters(model), lr=LEARNING_RATE, betas=ADAM_BETAS,
                                  eps=ADAM_EPSILON, weight_decay=0, foreach=False, fused=False)
    return Learner(model=model, optimizer=optimizer, evaluate=evaluate)


def evaluate(model, trajectory):
    features = trajectory.tokens[:, :1].float()
    return model(features).log_softmax(dim=-1)[:, 0]


def trajectory(name, probability=math.log(0.5)):
    request = Request(sample=name, group="group", prompt="reference", seed=0, limit=1, temperature=1)
    return Trajectory(request=request, tokens=torch.ones((1, 2), dtype=torch.long), prompt_length=1,
                      behavior=torch.tensor([probability], dtype=torch.float32), text="reference", truncated=False)


def batch(steps=(LOGICAL_ORDER,), advantages=ADVANTAGES, probability=math.log(0.5), reference=None, penalty=0, exchange=None,
          reference_source="engine", scored=None):
    trajectories = {name: trajectory(name, probability) for name in LOGICAL_ORDER}
    samples = {name: (tuple(words(item.behavior)), () if reference is None else (scalar.word(reference),), scalar.word(value))
               for (name, item), value in zip(trajectories.items(), advantages, strict=True)}
    core = Core(samples, steps, scalar.Profile(epsilon=0.2, penalty=penalty), reference_source=reference_source)
    return Plan(trajectories=trajectories, steps=steps, nonzero=sum(value != 0 for value in advantages),
                exchange=core if exchange is None else exchange, reference_source=reference_source,
                reference={} if scored is None else scored)


def run(plan):
    learner = make_learner()
    result = update(learner, plan)
    state = adapter_state(learner.model)
    optimizer = learner.optimizer.state_dict()
    gradients = tuple(value.grad.clone() for value in parameters(learner.model))
    return result.summary, state, optimizer, gradients, result.gradients


class LearningTests(unittest.TestCase):
    def test_one_step_reports_graph_values_and_applies_the_core_cotangents(self):
        learner = make_learner()
        logical = batch()
        before = digest(adapter_state(learner.model))
        with torch.no_grad():
            graph = {name: words(evaluate(learner.model, item)) for name, item in logical.trajectories.items()}
        result = update(learner, logical)
        core = logical.exchange
        currents = [record for record in core.records if record[0] == "current"]
        self.assertEqual([(step, name, observed, state) for _, step, name, observed, state in currents],
                         [(0, name, graph[name], before) for name in LOGICAL_ORDER])
        self.assertEqual(result.proximal, graph)
        self.assertEqual(result.currents, tuple((0, name, graph[name]) for name in LOGICAL_ORDER))
        self.assertEqual(core.records[-1], ("applied", 0, before, result.summary["after"], tuple(core.answered)))
        self.assertEqual(result.summary["before"], before)
        self.assertEqual(result.summary["after"], digest(adapter_state(learner.model)))
        self.assertNotEqual(result.summary["before"], result.summary["after"])

    def test_later_steps_report_proximal_first_and_chain_the_state(self):
        learner = make_learner()
        logical = batch(steps=(("a",), ("b", "c")))
        initial = digest(adapter_state(learner.model))
        with torch.no_grad():
            graph = {name: words(evaluate(learner.model, item)) for name, item in logical.trajectories.items()}
        result = update(learner, logical)
        records = logical.exchange.records
        self.assertEqual(records[:2], [("proximal", "b", graph["b"]), ("proximal", "c", graph["c"])])
        self.assertEqual(records[2][:4], ("current", 0, "a", graph["a"]))
        applied = [record for record in records if record[0] == "applied"]
        self.assertEqual([record[1] for record in applied], [0, 1])
        self.assertEqual(applied[0][2], initial)
        self.assertEqual(applied[1][2], applied[0][3])
        self.assertEqual(applied[1][3], result.summary["after"])
        later = [record for record in records if record[0] == "current" and record[1] == 1]
        self.assertEqual([record[4] for record in later], [applied[0][3]] * 2)
        self.assertNotEqual([record[3] for record in later], [graph["b"], graph["c"]])
        self.assertEqual(result.proximal, graph)

    def test_learner_scored_reference_words_come_from_the_reference_adapter(self):
        learner = make_learner()
        scored = {name: (torch.full_like(value, 0.5) if "lora_B" in name else value.clone())
                  for name, value in adapter_state(learner.model).items()}
        logical = batch(scored=scored, reference_source="learner")
        result = update(learner, logical)
        records = logical.exchange.records
        reported = {record[1]: record[2] for record in records if record[0] == "reference"}
        self.assertEqual(set(reported), set(LOGICAL_ORDER))
        self.assertEqual(result.reference, reported)
        expected = make_learner()
        set_peft_model_state_dict(expected.model, scored)
        policy = make_learner()
        with torch.no_grad():
            self.assertEqual(reported, {name: words(evaluate(expected.model, item)) for name, item in logical.trajectories.items()})
            policy_words = {name: words(evaluate(policy.model, item)) for name, item in logical.trajectories.items()}
        self.assertNotEqual(reported, policy_words)
        first = {record[2]: record[3] for record in records if record[0] == "current" and record[1] == 0}
        self.assertEqual(first, policy_words)

    def test_learner_reference_source_requires_the_frozen_adapter(self):
        with self.assertRaises(ValueError):
            update(make_learner(), batch(reference_source="learner"))

    def test_consumed_digests_are_the_cotangent_tensors_given_to_the_vjp(self):
        applied = []

        class Recorder(Core):
            def current(self, **values):
                objective, reward = super().current(**values)
                applied.append(observation(objective + reward))
                return objective, reward

        logical = batch()
        core = Recorder(logical.exchange.samples, logical.steps, logical.exchange.profile)
        update(make_learner(), replace(logical, exchange=core))
        self.assertEqual(core.records[-1][4], tuple(applied))

    def test_json_exchange_rejects_cotangents_for_another_step(self):
        answers = {"binding": {**BINDING, "attempt": 12}, "step": 1, "sample": "b", "observation": "0" * 64,
                   "state": "1" * 64}
        for name, value in [*answers.items(), ("objective", [0, 0]), ("reward", [-1]), ("stage", "current"), ("step", True)]:
            with self.subTest(field=name):
                learner = make_learner()
                before = adapter_state(learner.model)
                output = io.StringIO()
                logical = batch()
                core = logical.exchange
                forged = receiver(output, core)

                def receive():
                    reply = json.loads(forged())
                    reply[name] = value
                    return json.dumps(reply)

                exchange = Exchange(binding=BINDING, emit=report, receive=receive)
                with redirect_stdout(output), self.assertRaises(ValueError):
                    update(learner, replace(logical, exchange=exchange))
                assert_equal(before, adapter_state(learner.model))
                self.assertEqual(learner.optimizer.state_dict()["state"], {})

    def test_json_exchange_round_trips_the_reported_step(self):
        output = io.StringIO()
        logical = batch()
        core = logical.exchange
        exchange = Exchange(binding=BINDING, emit=report, receive=receiver(output, core))
        with redirect_stdout(output):
            result = update(make_learner(), replace(logical, exchange=exchange))
        emitted = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([value["stage"] for value in emitted], ["current"] * 3 + ["applied"])
        self.assertTrue(all(value["binding"] == BINDING for value in emitted))
        self.assertEqual([value["observation"] for value in emitted[:3]],
                         [observation(words) for _, _, words in result.currents])
        self.assertEqual(emitted[-1]["consumed"], core.answered)

    def test_scalar_failure_cannot_reach_model_backward_or_optimizer_step(self):
        learner = make_learner()
        before = adapter_state(learner.model)
        with self.assertRaisesRegex(ValueError, "Probability ratio underflow"):
            update(learner, batch(reference=math.log(0.5) - 1000))
        assert_equal(before, adapter_state(learner.model))
        self.assertEqual(learner.optimizer.state_dict()["state"], {})
        self.assertTrue(all(parameter.grad is None for parameter in parameters(learner.model)))

    def test_wrong_actual_dtype_fails_before_backward_or_optimizer_step(self):
        learner = make_learner()
        update(learner, batch())
        learner.optimizer.zero_grad(set_to_none=True)
        before = adapter_state(learner.model)
        slots = {parameter: {name: value.clone() for name, value in state.items()}
                 for parameter, state in learner.optimizer.state.items()}

        def wrong_dtype(model, trajectory):
            return evaluate(model, trajectory).double()

        with self.assertRaisesRegex(ValueError, "Learner graph values must be FP32 response vectors"):
            update(replace(learner, evaluate=wrong_dtype), batch())
        assert_equal(before, adapter_state(learner.model))
        for parameter, state in slots.items():
            assert_equal(state, learner.optimizer.state[parameter])
            self.assertIsNone(parameter.grad)

    def test_recorded_gradients_are_the_first_step_pre_step_snapshots(self):
        learner = make_learner()
        result = update(learner, batch())
        named = {name: parameter for name, parameter in learner.model.named_parameters() if parameter.requires_grad}
        self.assertEqual(set(result.gradients), {f"{role}/{name}" for role in ("objective", "reward") for name in named})
        for name, parameter in named.items():
            assert_equal(parameter.grad, result.gradients[f"objective/{name}"])
            assert_equal(parameter.grad, result.gradients[f"reward/{name}"])
        staged = make_learner()
        later = update(staged, batch(steps=(LOGICAL_ORDER, LOGICAL_ORDER)))
        assert_equal(later.gradients, result.gradients)
        self.assertNotEqual(later.summary["after"], result.summary["after"])

    def test_later_nonfinite_gradients_stop_before_that_optimizer_step(self):
        class Infinite(Core):
            def current(self, **values):
                objective, reward = super().current(**values)
                return ((0x7F800000,) * len(objective), reward) if values["step"] == 1 else (objective, reward)

        logical = batch(steps=(LOGICAL_ORDER, LOGICAL_ORDER))
        core = Infinite(logical.exchange.samples, logical.steps, logical.exchange.profile)
        learner = make_learner()
        with self.assertRaisesRegex(RuntimeError, "non-finite learner gradients"):
            update(learner, replace(logical, exchange=core))
        applied = [record for record in core.records if record[0] == "applied"]
        self.assertEqual(len(applied), 1)
        self.assertEqual(digest(adapter_state(learner.model)), applied[0][3])
        self.assertTrue(all(slot["step"].item() == 1 for slot in learner.optimizer.state.values()))

    def test_finite_gradients_cannot_accept_overflowed_optimizer_slots(self):
        learner = make_learner()
        with self.assertRaisesRegex(RuntimeError, "Non-finite optimizer state"):
            update(learner, batch(advantages=(OVERFLOW_ADVANTAGE,) * 3))
        self.assertTrue(all(value.isfinite().all() for value in parameters(learner.model)))
        self.assertTrue(all(value.grad.isfinite().all() for value in parameters(learner.model)))
        self.assertTrue(any(not slot.isfinite().all()
                            for slots in learner.optimizer.state.values() for slot in slots.values()))

    def test_nonfinite_existing_slots_are_rejected_before_another_step(self):
        learner = make_learner()
        update(learner, batch())
        slots = next(iter(learner.optimizer.state.values()))
        slots["exp_avg_sq"].fill_(float("inf"))
        before = adapter_state(learner.model)
        previous_step = slots["step"].clone()
        with self.assertRaisesRegex(RuntimeError, "Non-finite optimizer state"):
            update(learner, batch())
        assert_equal(before, adapter_state(learner.model))
        assert_equal(previous_step, slots["step"])

    def test_small_advantage_reaches_the_reward_gradient(self):
        expected = run(batch())
        analytic_norm = SMALL_ADVANTAGE * math.sqrt(2) / (2 * len(LOGICAL_ORDER))
        self.assertAlmostEqual(expected[0]["reward_gradient_norm"], analytic_norm, places=15)
        self.assertGreater(expected[0]["reward_gradient_norm"], 0)
        self.assertEqual(expected[0]["active_tokens"], len(LOGICAL_ORDER))
        self.assertEqual(expected[0]["nonzero_advantages"], len(LOGICAL_ORDER))

    def test_declared_step_order_is_the_accumulation_order(self):
        original, reordered = run(batch()), run(batch(steps=(("a", "c", "b"),)))
        self.assertNotEqual(digest(original[1]), digest(reordered[1]))
        self.assertNotEqual(digest(original[4]), digest(reordered[4]))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
