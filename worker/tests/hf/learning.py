import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import math
import unittest
from dataclasses import replace
from itertools import permutations

import torch
from peft import LoraConfig, get_peft_model

from worker.hf.learning import Batch, Learner, Sample, parameters, probabilities, update
from worker.hf.objective import Profile
from worker.hf.probe import ADAM_BETAS, ADAM_EPSILON, LEARNING_RATE, adapter_state, assert_equal, digest
from worker.hf.probability import words
from worker.hf.rollout import Request, Trajectory
from worker.record import ROLES

SMALL_ADVANTAGE = 2 ** -28
OVERFLOW_ADVANTAGE = 1e30
LOGICAL_ORDER = ("a", "b", "c")


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


def sample(name, advantage):
    request = Request(sample=name, group="group", prompt="reference", seed=0, limit=1, temperature=1)
    probability = torch.tensor([math.log(0.5)], dtype=torch.float32)
    trajectory = Trajectory(request=request, tokens=torch.ones((1, 2), dtype=torch.long),
                            prompt_length=1, behavior=probability, text="reference", truncated=False)
    return Sample(trajectory=trajectory, proximal=probability, reference=probability, advantage=advantage)


def batch():
    samples = tuple(sample(name, value) for name, value in zip(LOGICAL_ORDER, (1, -1, SMALL_ADVANTAGE)))
    return Batch(samples=samples, order=LOGICAL_ORDER, profile=Profile(epsilon=0.2, penalty=0))


def run(delivered):
    learner = make_learner()
    result = update(learner, delivered)
    state = adapter_state(learner.model)
    optimizer = learner.optimizer.state_dict()
    gradients = tuple(value.grad.clone() for value in parameters(learner.model))
    return result.summary, state, optimizer, gradients, result.gradients


class LearningTests(unittest.TestCase):
    def test_identical_reference_adapter_reuses_the_proximal_observation(self):
        learner = make_learner()
        trajectories = tuple(item.trajectory for item in batch().samples)
        current = adapter_state(learner.model)
        proximal, fixed = probabilities(learner.model, trajectories, current, evaluate=evaluate)
        self.assertIs(fixed, proximal)
        shifted = {name: value + torch.arange(1, value.numel() + 1, dtype=value.dtype).reshape(value.shape)
                   if "lora_B" in name else value for name, value in current.items()}
        repeated, distinct = probabilities(learner.model, trajectories, shifted, evaluate=evaluate)
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(proximal, repeated, strict=True)))
        self.assertTrue(all(not torch.equal(old, new) for old, new in zip(proximal, distinct, strict=True)))
        assert_equal(current, adapter_state(learner.model))

    def test_objective_roles_are_the_behavior_words_and_the_graph_is_recorded(self):
        learner = make_learner()
        sampled = torch.tensor([math.log(0.25)], dtype=torch.float32)
        logical = batch()
        logical = replace(logical, samples=tuple(replace(item, trajectory=replace(item.trajectory, behavior=sampled),
                                                         proximal=sampled, reference=sampled) for item in logical.samples))
        with torch.no_grad():
            graph = [words(evaluate(learner.model, item.trajectory)) for item in logical.samples]
        result = update(learner, logical)
        for observed, item, linearized in zip(result.probabilities, logical.samples, graph, strict=True):
            for role in ("behavior", "proximal", "current"):
                self.assertEqual(observed.words[ROLES.index(role)], words(item.trajectory.behavior))
            self.assertEqual(observed.linearized, linearized)
            self.assertNotEqual(observed.linearized, words(item.trajectory.behavior))

    def test_scalar_failure_cannot_reach_model_backward_or_optimizer_step(self):
        learner = make_learner()
        before = adapter_state(learner.model)

        logical = batch()
        underflow = replace(logical, samples=tuple(replace(item, reference=item.reference - 1000) for item in logical.samples))
        with self.assertRaisesRegex(ValueError, "Probability ratio underflow"):
            update(learner, underflow)
        assert_equal(before, adapter_state(learner.model))
        self.assertEqual(learner.optimizer.state_dict()["state"], {})
        self.assertTrue(all(parameter.grad is None for parameter in parameters(learner.model)))

    def test_wrong_actual_dtype_fails_before_backward_or_optimizer_step(self):
        learner = make_learner()
        logical = batch()
        update(learner, logical)
        learner.optimizer.zero_grad(set_to_none=True)
        before = adapter_state(learner.model)
        slots = {parameter: {name: value.clone() for name, value in state.items()}
                 for parameter, state in learner.optimizer.state.items()}

        def wrong_dtype(model, trajectory):
            return evaluate(model, trajectory).double()

        with self.assertRaisesRegex(ValueError, "Learner graph values must be FP32 response vectors"):
            update(replace(learner, evaluate=wrong_dtype), logical)
        assert_equal(before, adapter_state(learner.model))
        for parameter, state in slots.items():
            assert_equal(state, learner.optimizer.state[parameter])
            self.assertIsNone(parameter.grad)

    def test_recorded_gradients_are_named_pre_step_snapshots(self):
        learner = make_learner()
        result = update(learner, batch())
        named = {name: parameter for name, parameter in learner.model.named_parameters() if parameter.requires_grad}
        self.assertEqual(set(result.gradients), {f"{role}/{name}" for role in ("objective", "reward") for name in named})
        for name, parameter in named.items():
            assert_equal(parameter.grad, result.gradients[f"objective/{name}"])
            assert_equal(parameter.grad, result.gradients[f"reward/{name}"])
        recorded = {name: value.clone() for name, value in result.gradients.items()}
        for parameter in parameters(learner.model):
            parameter.grad.zero_()
        assert_equal(recorded, result.gradients)

    def test_finite_gradients_cannot_accept_overflowed_optimizer_slots(self):
        logical = batch()
        large = tuple(replace(item, advantage=OVERFLOW_ADVANTAGE) for item in logical.samples)
        learner = make_learner()
        with self.assertRaisesRegex(RuntimeError, "Non-finite optimizer state"):
            update(learner, replace(logical, samples=large))
        self.assertTrue(all(value.isfinite().all() for value in parameters(learner.model)))
        self.assertTrue(all(value.grad.isfinite().all() for value in parameters(learner.model)))
        self.assertTrue(any(not slot.isfinite().all()
                            for slots in learner.optimizer.state.values() for slot in slots.values()))

    def test_nonfinite_existing_slots_are_rejected_before_another_step(self):
        learner = make_learner()
        logical = batch()
        update(learner, logical)
        slots = next(iter(learner.optimizer.state.values()))
        slots["exp_avg_sq"].fill_(float("inf"))
        before = adapter_state(learner.model)
        previous_step = slots["step"].clone()
        with self.assertRaisesRegex(RuntimeError, "Non-finite optimizer state"):
            update(learner, logical)
        assert_equal(before, adapter_state(learner.model))
        assert_equal(previous_step, slots["step"])

    def test_delivery_order_does_not_change_actual_update(self):
        logical = batch()
        expected = run(logical)
        analytic_norm = SMALL_ADVANTAGE * math.sqrt(2) / (2 * len(LOGICAL_ORDER))
        self.assertAlmostEqual(expected[0]["reward_gradient_norm"], analytic_norm, places=15)
        self.assertGreater(expected[0]["reward_gradient_norm"], 0)
        self.assertNotEqual(expected[0]["before"], expected[0]["after"])
        for delivery in permutations(logical.samples):
            with self.subTest(order=tuple(item.trajectory.request.sample for item in delivery)):
                actual = run(replace(logical, samples=delivery))
                assert_equal(expected[1:], actual[1:])
                self.assertEqual(expected[0], actual[0])

    def test_algorithm_order_is_not_delivery_order(self):
        logical = batch()
        changed = replace(logical, order=("a", "c", "b"))
        original, reordered = run(logical), run(changed)
        self.assertNotEqual(digest(original[1]), digest(reordered[1]))
        self.assertNotEqual(digest(original[4]), digest(reordered[4]))

    def test_malformed_batch_is_rejected_before_optimizer_changes(self):
        logical = batch()
        invalid = [replace(logical, order=()), replace(logical, order=("a", "a", "b")),
                   replace(logical, order=("a", "b", "")),
                   replace(logical, order=("a", "b", "missing")),
                   replace(logical, samples=logical.samples[:-1]),
                   replace(logical, samples=logical.samples + logical.samples[:1])]
        for candidate in invalid:
            with self.subTest(order=candidate.order):
                learner = make_learner()
                before = adapter_state(learner.model)
                with self.assertRaises(ValueError):
                    update(learner, candidate)
                assert_equal(before, adapter_state(learner.model))
                self.assertEqual(learner.optimizer.state_dict()["state"], {})


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
