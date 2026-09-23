import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

from dataclasses import replace

import torch

from worker.hf.learning import accumulate_objective, parameter_vjps, update
from worker.hf.probe import adapter_state
from worker.hf.tensors import assert_equal
from worker.tests.hf.learning import batch, make_learner

REFERENCE_LOG_SHIFT = 0.125
PENALTY = 0.04


class BatchedRoleTests(unittest.TestCase):
    def test_distinct_adjoints_match_independent_polynomial_derivatives(self):
        first = torch.tensor(1.25, requires_grad=True)
        second = torch.tensor(-0.75, requires_grad=True)
        current = torch.stack((first.square() + second, first * second + second.square()))
        objective, reward = parameter_vjps(current, (first, second),
                                          objective=torch.tensor((0.25, -0.5)),
                                          reward=torch.tensor((0.75, 0.125)))
        assert_equal(torch.stack(objective), torch.tensor((1.0, 0.375)))
        assert_equal(torch.stack(reward), torch.tensor((1.78125, 0.71875)))
        self.assertIsNone(first.grad)
        self.assertIsNone(second.grad)
        accumulate_objective((first, second), objective)
        assert_equal(torch.stack((first.grad, second.grad)), torch.tensor((1.0, 0.375)))

    def test_nonzero_penalty_records_both_roles_and_adam_consumes_objective(self):
        learner, expected = make_learner(), make_learner()
        logical = batch()
        samples = tuple(replace(item, reference=item.reference - REFERENCE_LOG_SHIFT) for item in logical.samples)
        delivered = replace(logical, samples=samples, profile=replace(logical.profile, penalty=PENALTY))
        result = update(learner, delivered)
        named = {name: parameter for name, parameter in expected.model.named_parameters() if parameter.requires_grad}
        self.assertTrue(any(not torch.equal(result.gradients[f"objective/{name}"], result.gradients[f"reward/{name}"])
                            for name in named))
        self.assertGreater(result.summary["reward_gradient_norm"], 0)
        self.assertGreater(result.summary["gradient_norm"], 0)
        for name, parameter in named.items():
            parameter.grad = result.gradients[f"objective/{name}"].clone()
        expected.optimizer.step()
        assert_equal(adapter_state(learner.model), adapter_state(expected.model))
        assert_equal(learner.optimizer.state_dict(), expected.optimizer.state_dict())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
