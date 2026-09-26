import unittest

try:
    import torch  # noqa: F401
    import vllm  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

from worker.trajectory import Request
from worker.vllm.mapping import Selection
from worker.vllm.state import Binding, Monitor


class Native:
    lora_index_to_id = ()


def bound():
    request = Request(sample="s0", group="test/0", prompt="p", seed=17, limit=8, temperature=0.8)
    return Binding(internal="r0", external="e0", request=request, prompt=(1, 2, 3), eos=4,
                   selection=Selection(adapter=1, description="d"), receipt="package")


def monitor(*, models=None, observations=None):
    calls = [] if observations is None else observations
    return Monitor(object(), Native(), bindings=(bound(),), verify=lambda **fields: fields["receipt"],
                   observe_model=lambda: calls.append(1) or ("observed",), models=models), calls


class PermissionTests(unittest.TestCase):
    def test_execution_permission_is_consumed_once(self):
        state, _ = monitor()
        state.allow()
        with self.assertRaisesRegex(RuntimeError, "permission"):
            state.allow()

    def test_model_step_requires_granted_permission(self):
        state, _ = monitor()
        with self.assertRaisesRegex(RuntimeError, "approved"):
            state.before_model({"input_ids": torch.tensor([1, 2, 3]), "inputs_embeds": None})

    def test_supplied_observation_is_used_without_observing_again(self):
        state, calls = monitor(models=("supplied",))
        self.assertEqual(state.models, ("supplied",))
        self.assertEqual(calls, [])

    def test_absent_observation_is_taken_exactly_once(self):
        state, calls = monitor()
        self.assertEqual(state.models, ("observed",))
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()
