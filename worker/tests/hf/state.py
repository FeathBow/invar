import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

import copy
import unittest

import torch
from peft.tuners.lora.layer import LoraLayer

from worker.hf import assembly
from worker.hf import runtime as learner_runtime
from worker.hf import operation
from worker.hf.probe import assert_equal
from worker.hf.tensors import fingerprint
from worker.tests.hf import resident as Fixture


def snapshot(runtime):
    return copy.deepcopy((runtime.learner.model.state_dict(), runtime.learner.optimizer.state_dict(),
                          runtime.reference, torch.get_rng_state(), torch.cuda.get_rng_state_all(),
                          [module.training for module in runtime.learner.model.modules()],
                          [value.grad for value in runtime.learner.model.parameters()],
                          operation.digest(runtime.tokenizer), assembly.digest(runtime.learner.model)))


def mutate(runtime, field):
    model, optimizer = runtime.learner.model, runtime.learner.optimizer
    if field in ("policy", "base"):
        next(value for value in model.parameters() if value.requires_grad == (field == "policy")).add_(1)
    elif field in ("exp_avg", "exp_avg_sq", "step"):
        next(iter(optimizer.state.values()))[field].add_(1)
    elif field == "optimizer":
        optimizer.param_groups[0]["lr"] *= 2
    elif field == "reference":
        next(iter(runtime.reference.values())).add_(1)
    elif field == "tokenizer":
        runtime.tokenizer.add_tokens(["changed unused tokenizer entry"])
    elif field == "assembly":
        next(module for module in model.modules() if isinstance(module, LoraLayer)).scaling["default"] += 1
    elif field == "rng":
        torch.manual_seed(999)
    elif field == "mode":
        model.train()
    elif field == "gradient":
        value = next(value for value in model.parameters() if value.requires_grad)
        value.grad = torch.ones_like(value)
    else:
        raise ValueError("Unknown resident mutation")


class LearnerStateTests(unittest.TestCase):
    def test_fingerprint_preserves_types_shapes_and_floating_point_words(self):
        original = {"optimizer": [torch.tensor([-0.0, 1.0])], "rate": -0.0, "parameters": {0: "adapter"}}
        self.assertEqual(fingerprint(original), fingerprint(dict(reversed(list(original.items())))))
        changed = [{**original, "optimizer": (original["optimizer"][0],)},
                   {**original, "rate": 0.0}, {**original, "parameters": {"0": "adapter"}},
                   {**original, "optimizer": [torch.tensor([0.0, 1.0])]},
                   {**original, "optimizer": [original["optimizer"][0].double()]},
                   {**original, "optimizer": [original["optimizer"][0].reshape(1, 2)]}]
        for value in changed:
            self.assertNotEqual(fingerprint(value), fingerprint(original))
        with self.assertRaisesRegex(TypeError, "Unsupported checkpoint"):
            fingerprint({"unsupported": object()})

    def test_changed_live_state_is_rejected_and_never_silently_restored(self):
        fields = ("policy", "base", "exp_avg", "exp_avg_sq", "step", "optimizer",
                  "reference", "tokenizer", "assembly", "rng", "mode", "gradient")
        for field in fields:
            with self.subTest(field=field):
                runtime, call, paths, observations = Fixture.prepare()
                runtime = Fixture.execute(runtime, call, paths, observations)
                following = Fixture.next_call(runtime, call)
                with torch.no_grad():
                    mutate(runtime, field)
                before = snapshot(runtime)
                with self.assertRaises(RuntimeError):
                    learner_runtime.activate(runtime, following.request)
                assert_equal(before, snapshot(runtime))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
