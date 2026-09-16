import unittest

try:
    import torch  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

from dataclasses import replace
from contextlib import redirect_stdout
from functools import partial
import io
import json
import shutil
from types import SimpleNamespace
import unittest

import torch
from safetensors.torch import load_file

from worker.advantage import check
from worker import binding
from worker.invocation import approve
from worker.hf import runtime as learner_runtime
from worker.hf.probe import adapter_state, assert_equal
from worker.hf.rollout import logprobs
from worker.hf import step
from worker.tests.hf.inference import IDENTITY, measured, model
from worker.tests.hf.learner import fixture
from worker.tests.hf.handshake import cpu_measure
from worker.tests.hf.tokenization import make_tokenizer


def permission(bound):
    source = io.StringIO(json.dumps({"binding": bound.binding(), "program": bound.program}) + "\n")
    approve(bound, source=source)


def prepare():
    call, paths = fixture()
    output = io.StringIO()
    emit = lambda stage, values: output.write(json.dumps({"stage": stage, **values}) + "\n")
    measure = partial(cpu_measure, emit=emit)

    def loader(options, request):
        return measure("load", model), make_tokenizer(), IDENTITY

    runtime, loaded = learner_runtime.initialize(paths, call.request, loader=loader,
                                                  measure=measure, evaluate=partial(logprobs, device="cpu"))
    return runtime, call, paths, (output, emit, measure, loaded)


def execute(runtime, call, paths, observations, *, permit=permission):
    _, emit, measure, loaded = observations
    paths.output.mkdir()
    updated = learner_runtime.execute(runtime, call, paths.output, loaded=loaded,
                                        checked=check(call.request), measure=measure, permission=permit, emit=emit)
    measure("released", partial(learner_runtime.release, updated))
    return updated


def next_call(runtime, previous):
    bound = previous.invocation
    next_binding = {name: getattr(bound, name) + 100 for name in ("call", "attempt", "instance")}
    return replace(previous, invocation=replace(bound, **next_binding), load=replace(previous.load, **next_binding),
                   request=replace(previous.request, policy=runtime.saved.policy, learner=runtime.saved.learner))


class LearnerResidentTests(unittest.TestCase):
    def test_actual_continuation_preserves_objects_and_matches_fresh_checkpoint_restore(self):
        runtime, call, paths, observations = prepare()
        model_id, optimizer_id = id(runtime.learner.model), id(runtime.learner.optimizer)
        pointers = {name: value.data_ptr() for name, value in runtime.learner.model.named_parameters()}
        runtime = execute(runtime, call, paths, observations)
        self.assertNotEqual(runtime.saved.policy, call.request.policy)
        baseline_checkpoint = paths.output.parent / "baseline-checkpoint"
        shutil.copytree(paths.output, baseline_checkpoint)
        baseline_reference = paths.output.parent / "baseline-reference.safetensors"
        shutil.copyfile(paths.reference, baseline_reference)
        following = next_call(runtime, call)
        for path in (paths.checkpoint / "adapter.safetensors", paths.checkpoint / "learner.pt",
                     paths.output / "adapter.safetensors", paths.output / "learner.pt"):
            path.write_bytes(b"changed source; resident state must not be restored or reread")
        loaded = learner_runtime.activate(runtime, following.request)
        next_paths = SimpleNamespace(**{**vars(paths), "checkpoint": paths.output,
                                        "output": paths.output.parent / "resident-second"})
        runtime = execute(runtime, following, next_paths, (*observations[:3], loaded))
        self.assertEqual((id(runtime.learner.model), id(runtime.learner.optimizer)), (model_id, optimizer_id))
        self.assertEqual({name: value.data_ptr() for name, value in runtime.learner.model.named_parameters()}, pointers)
        self.assertTrue(all(value.grad is None for value in runtime.learner.model.parameters()))
        self.assertTrue(all(slot["step"].item() == 2 for slot in runtime.learner.optimizer.state.values()))
        baseline_paths = SimpleNamespace(**{**vars(next_paths), "checkpoint": baseline_checkpoint,
                                            "reference": baseline_reference, "output": paths.output.parent / "fresh-second"})
        with redirect_stdout(io.StringIO()):
            step.run(following, baseline_paths, loader=lambda options, request: (model(), make_tokenizer(), IDENTITY),
                     measure=measured, permission=permission, evaluate=partial(logprobs, device="cpu"))
        assert_equal(adapter_state(runtime.learner.model), load_file(baseline_paths.output / "adapter.safetensors"))
        assert_equal(load_file(next_paths.output / "gradients.safetensors"), load_file(baseline_paths.output / "gradients.safetensors"))
        self.assertEqual((next_paths.output / "probabilities.json").read_bytes(), (baseline_paths.output / "probabilities.json").read_bytes())
        saved = [torch.load(root / "learner.pt", weights_only=True) for root in (next_paths.output, baseline_paths.output)]
        assert_equal(binding.observation(saved[0]), binding.observation(saved[1]))
        results = [value for line in observations[0].getvalue().splitlines() if (value := json.loads(line))["stage"] == "result"]
        self.assertEqual(len(results), 2)
        self.assertTrue(all(value["update"]["reward_gradient_norm"] > 0 for value in results))

    def test_permission_time_state_change_prevents_another_numerical_forward(self):
        runtime, call, paths, observations = prepare()
        evaluated = []

        def evaluate(model, trajectory):
            evaluated.append(trajectory.request.sample)
            return logprobs(model, trajectory, device="cpu")

        runtime = replace(runtime, learner=replace(runtime.learner, evaluate=evaluate))

        def changed(bound):
            permission(bound)
            with torch.no_grad():
                next(value for value in runtime.learner.model.parameters() if value.requires_grad).add_(1)

        with self.assertRaisesRegex(RuntimeError, "state differs from the saved checkpoint"):
            execute(runtime, call, paths, observations, permit=changed)
        self.assertEqual(call.request.reference, call.request.policy)
        self.assertEqual(len(evaluated), len(call.request.samples))
        self.assertEqual(list(paths.output.iterdir()), [])
        self.assertNotIn('"stage": "reward_update"', observations[0].getvalue())
        self.assertNotIn('"stage": "result"', observations[0].getvalue())

    def test_next_update_requires_the_actual_completed_successor_and_fixed_reference(self):
        runtime, call, paths, observations = prepare()
        runtime = execute(runtime, call, paths, observations)
        following = next_call(runtime, call)
        for field in ("policy", "learner", "reference", "tokenizer", "base", "assembly"):
            with self.subTest(field=field), self.assertRaises((ValueError, RuntimeError)):
                learner_runtime.activate(runtime, replace(following.request, **{field: "0" * 64}))
        original = adapter_state(runtime.learner.model)
        wrong = replace(following.request.samples[0], text="a different decoded response")
        with self.assertRaisesRegex(ValueError, "Observed text differs"):
            learner_runtime.activate(runtime, replace(following.request, samples=(wrong, *following.request.samples[1:])))
        assert_equal(original, adapter_state(runtime.learner.model))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
