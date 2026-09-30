import unittest

try:
    import mlx  # noqa: F401
except ImportError as missing:
    raise unittest.SkipTest(f"{missing.name} is not installed") from missing

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import weakref

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from worker.tests.advantage import Reward, advantages
from worker.cohort import Optimizer
from worker import core
from worker.logical import Learner, Plan
from worker.mlx import checkpoint as mlx_checkpoint
from worker.mlx import codec as mlx_codec
from worker.mlx import learning as mlx_learning
from worker.tests.mlx import derivative
from worker.mlx import probability as mlx_probability
from worker.mlx import state as mlx_state
from worker.mlx import tensors as mlx_tensors
from worker import record as probability_record
from worker.tests import scalar
from worker.tests.responder import Core
from worker.trajectory import Request, Trajectory
from worker.update import observation

SETTINGS = Optimizer(learning_rate=0.0001, betas=(0.9, 0.999), epsilon=1e-8, weight_decay=0.0)
REWARDS = (Reward(sample="a", group="group", value=1.0), Reward(sample="b", group="group", value=0.0))
DELTA = 0.0001
IDENTITIES = {"tokenizer": "a" * 64, "base": "b" * 64, "assembly": "c" * 64}


class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_a = mx.array([[1.0]], dtype=mx.float32)
        self.lora_b = mx.zeros((1, 2), dtype=mx.float32)

    def __call__(self, values):
        return values @ self.lora_a @ self.lora_b


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = Projection()

    def __call__(self, values):
        return self.projection(values)


def evaluate(model, trajectory):
    logits = model(trajectory.tokens[:, :1].astype(mx.float32))
    return (logits - mx.logsumexp(logits, axis=-1, keepdims=True))[:, 0]


def learner():
    return Learner(model=Model(), optimizer=optim.AdamW(learning_rate=SETTINGS.learning_rate, betas=SETTINGS.betas,
                                                       eps=SETTINGS.epsilon, weight_decay=SETTINGS.weight_decay,
                                                       bias_correction=True), evaluate=evaluate)


def batch(model, *, reference=None):
    normalized = dict(advantages(REWARDS, DELTA))
    trajectories, samples = {}, {}
    for index, reward in enumerate(REWARDS, 1):
        request = Request(sample=reward.sample, group=reward.group, prompt="native derivative fixture",
                          seed=index, limit=1, temperature=1.0)
        tokens = mx.array([[index, 0]], dtype=mx.int32)
        initial = Trajectory(request=request, tokens=tokens, prompt_length=1,
                             behavior=mx.zeros((1,), dtype=mx.float32), text="fixture", truncated=True)
        probability = evaluate(model, initial)
        mx.eval(probability)
        trajectories[reward.sample] = Trajectory(request=request, tokens=tokens, prompt_length=1, behavior=probability,
                                                 text=initial.text, truncated=True)
        samples[reward.sample] = (mlx_probability.words(probability), () if reference is None else reference[index - 1],
                                  scalar.word(normalized[reward.sample]))
    order = tuple(reward.sample for reward in REWARDS)
    return Plan(trajectories=trajectories, steps=(order,), nonzero=sum(value != 0 for value in normalized.values()),
                exchange=Core(samples, (order,), scalar.Profile(epsilon=0.2, penalty=0.01)))


def again(plan):
    core = plan.exchange
    return replace(plan, exchange=Core(core.samples, core.steps, core.profile))


def save(directory, actual):
    directory.mkdir()
    parameters = mlx_learning.adapter(actual.model)
    policy = mlx_tensors.save_policy(directory / "adapter.safetensors", parameters)
    snapshot = mlx_checkpoint.observe(actual.optimizer, identities={"adapter": policy, **IDENTITIES}, parameters=parameters)
    mlx_checkpoint.save(directory / "learner.pt", snapshot)
    return policy, hashlib.sha256((directory / "learner.pt").read_bytes()).hexdigest()


def report(directory, result, logical, *, binding, input_state, reference):
    order = logical.steps[0]
    optimizer = {"learning_rate": SETTINGS.learning_rate, "betas": SETTINGS.betas,
                 "epsilon": SETTINGS.epsilon, "weight_decay": SETTINGS.weight_decay}
    request = {"specification": "grpo-token-mean/v1", "policy": input_state[0], "learner": input_state[1],
               "reference": reference, **IDENTITIES, "behavior_model": {name: IDENTITIES[name] for name in ("base", "assembly")},
               "optimizer": optimizer, "schedule": {"update": 0, "staleness": 0},
               "samples": [observation(logical.trajectories[reward.sample],
                                       SimpleNamespace(sample=reward.sample, group=reward.group, reward=reward.value, version=0,
                                                       behavior_policy=input_state[0],
                                                       reference_bits=() if reference == input_state[0] else logical.exchange.samples[reward.sample][1],
                                                       advantage_bits=logical.exchange.samples[reward.sample][2]))
                           for reward in REWARDS],
               "order": list(order), "steps": [list(batch) for batch in logical.steps],
               "epsilon": logical.exchange.profile.epsilon, "penalty": logical.exchange.profile.penalty, "delta": DELTA}
    bound = {"call": binding, "attempt": binding, "instance": binding}
    program = "native checkpoint association fixture; no execution-authority claim"
    gradients = directory / "gradients.safetensors"
    with gradients.open("xb") as target:
        mx.save_safetensors(target, result.gradients, metadata={"binding": json.dumps(bound), "program": program,
                                                               "policy": input_state[0], "observation": "objective and reward gradients before AdamW"})
    probability = probability_record.save(directory / "probabilities.json", order, result,
                                           invocation={"binding": bound, "program": program}, request=request)
    policy = mlx_tensors.digest(mlx_tensors.policy(directory / "adapter.safetensors", result.summary["after"]))
    finished = {"stage": "result", "binding": bound, "request": request, "update": result.summary, "adapter": policy,
                "learner": hashlib.sha256((directory / "learner.pt").read_bytes()).hexdigest(),
                "gradients": hashlib.sha256(gradients.read_bytes()).hexdigest(), "probabilities": probability,
                "storage": "staged; not published"}
    log = directory / "output.jsonl"
    consumed = {"stage": "consumed", "binding": bound, "program": program, "request": request}
    log.write_text("\n".join(json.dumps(value) for value in (consumed, finished)) + "\n")
    return log


class LearningTests(unittest.TestCase):
    def test_actual_parameter_vjp_and_saved_two_step_continuation(self):
        mx.random.seed(17)
        mx.eval(mx.random.uniform(shape=(3,)))
        directory = Path(tempfile.mkdtemp(prefix="invar-mlx-learning-"))
        live = learner()
        first = batch(live.model)
        reference = mlx_tensors.digest(mlx_learning.adapter(live.model))
        result = mlx_learning.update(live, first, linearize=derivative.linearize)
        quarter = scalar.word(scalar.rounded(scalar.number(first.exchange.samples["a"][2])) / 4)
        self.assertEqual(mlx_probability.words(result.gradients["reward/projection.lora_b"].reshape(-1)),
                         (quarter, scalar.word(-scalar.number(quarter))))
        self.assertNotEqual(result.summary["before"], result.summary["after"])
        saved = save(directory / "first", live)
        checkpoint = mlx_checkpoint.load((directory / "first/learner.pt").read_bytes())
        self.assertNotEqual(checkpoint["rng"][0].tolist(), mx.random.key(17).tolist())
        self.assertEqual(result.proximal, {name: values[0] for name, values in first.exchange.samples.items()})
        second = batch(live.model, reference=tuple(values[0] for values in first.exchange.samples.values()))
        continued = mlx_learning.update(live, second, linearize=derivative.linearize)
        save(directory / "live", live)
        restored = learner()
        policy = mlx_tensors.policy(directory / "first/adapter.safetensors", saved[0])
        mlx_state.restore(restored, checkpoint, policy=policy, identities={"adapter": saved[0], **IDENTITIES}, settings=SETTINGS)
        repeated = mlx_learning.update(restored, again(second), linearize=derivative.linearize)
        save(directory / "restored", restored)
        self.assertEqual(continued.summary, repeated.summary)
        self.assertTrue(mlx_tensors.equal(continued.gradients, repeated.gradients))
        self.assertEqual(live.optimizer.step.item(), 2)
        self.compare(directory, (continued, repeated), second, input_state=saved, reference=reference)
        self.random_continuation(restored, checkpoint, policy, saved[0])

    def test_later_steps_report_proximal_first_and_chain_the_state(self):
        live = learner()
        planned = batch(live.model)
        planned = replace(planned, steps=(("a",), ("b",)),
                          exchange=Core(planned.exchange.samples, (("a",), ("b",)), planned.exchange.profile))
        initial = mlx_tensors.digest(mlx_learning.adapter(live.model))
        result = mlx_learning.update(live, planned, linearize=derivative.linearize)
        records = planned.exchange.records
        self.assertEqual([record[:2] for record in records], [("proximal", "b"), ("current", 0), ("applied", 0), ("current", 1), ("applied", 1)])
        self.assertEqual(records[1][4], initial)
        self.assertEqual((records[2][2], records[3][4], records[4][2]), (initial, records[2][3], records[2][3]))
        self.assertEqual(records[4][3], result.summary["after"])
        self.assertEqual(records[4][4], tuple(planned.exchange.answered[1:]))
        self.assertNotEqual(records[3][3], records[0][2])
        self.assertEqual(live.optimizer.step.item(), 2)

    def test_proximal_callbacks_are_released_before_the_first_step(self):
        live = learner()
        planned = batch(live.model)
        steps = (("a",), ("b",))
        released = []

        def linearize(model, trajectory, *, evaluate):
            current, differentiate = derivative.linearize(model, trajectory, evaluate=evaluate)
            released.append(weakref.ref(differentiate))
            return current, differentiate

        class Checked(Core):
            def current(self, **values):
                if values["step"] == 0:
                    self.proximal_alive = released[0]() is not None
                return super().current(**values)

        core = Checked(planned.exchange.samples, steps, planned.exchange.profile)
        mlx_learning.update(live, replace(planned, steps=steps, exchange=core), linearize=linearize)
        self.assertFalse(core.proximal_alive)

    def compare(self, directory, results, logical, *, input_state, reference):
        arguments = ["compare", "states", "--codec-mode", "stdio", "--policy", directory / "first/adapter.safetensors"]
        for side, name, result, call in zip(("left", "right"), ("live", "restored"), results, (7, 8), strict=True):
            output = directory / name
            log = report(output, result, logical, binding=call, input_state=input_state, reference=reference)
            arguments.extend(("--" + side + "-checkpoint", output, "--" + side + "-log", log, "--" + side + "-call", call))
        observed = core.exchange(arguments, executable=os.environ.get("INVAR_CORE", "invar"), handler=mlx_codec.Session().handle)
        self.assertTrue(observed["equal"])

    def random_continuation(self, actual, checkpoint, policy, identity):
        expected = mx.random.uniform(shape=(4,))
        mx.eval(expected)
        mlx_state.restore(actual, checkpoint, policy=policy, identities={"adapter": identity, **IDENTITIES}, settings=SETTINGS)
        observed = mx.random.uniform(shape=(4,))
        self.assertEqual(mlx_probability.words(expected), mlx_probability.words(observed))


if __name__ == "__main__":
    unittest.main()
