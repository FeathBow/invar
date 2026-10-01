import json

from worker.tests import scalar
from worker.exchange import observation


class Core:
    def __init__(self, samples, steps, profile, *, binding=None, reference_source="engine"):
        self.samples, self.steps, self.profile, self.binding = samples, steps, profile, binding
        self.reference_source = reference_source
        self.fixed, self.records, self.answered, self.scored = {}, [], [], {}

    def proximal(self, *, sample, words):
        self.records.append(("proximal", sample, tuple(words)))
        self.fixed[sample] = tuple(words)

    def reference(self, *, sample, words):
        self.records.append(("reference", sample, tuple(words)))
        self.scored[sample] = tuple(words)

    def current(self, *, step, sample, words, state):
        self.records.append(("current", step, sample, tuple(words), state))
        if step == 0:
            self.fixed[sample] = tuple(words)
        behavior, reference, advantage = self.samples[sample]
        chosen = self.scored[sample] if self.reference_source == "learner" else (reference or behavior)
        total = sum(len(self.samples[name][0]) for name in self.steps[step])
        inputs = tuple(scalar.Inputs(current=now, proximal=old, behavior=seen, reference=fixed, advantage=advantage)
                       for now, old, seen, fixed in zip(words, self.fixed[sample], behavior, chosen, strict=True))
        outputs = scalar.calculate(self.profile, total, inputs)
        objective, reward = tuple(item.gradient for item in outputs), tuple(item.reward_gradient for item in outputs)
        self.answered.append(observation(objective + reward))
        return objective, reward

    def applied(self, *, step, before, after, consumed):
        self.records.append(("applied", step, before, after, tuple(consumed)))

    def reply(self, line):
        value = json.loads(line)
        if value.get("stage") == "proximal":
            self.proximal(sample=value["sample"], words=value["words"])
            return None
        if value.get("stage") == "applied":
            self.applied(step=value["step"], before=value["before"], after=value["after"], consumed=value["consumed"])
            return None
        if value.get("stage") != "current":
            return None
        objective, reward = self.current(step=value["step"], sample=value["sample"], words=value["words"], state=value["state"])
        answer = {"stage": "cotangents", "binding": value["binding"] if self.binding is None else self.binding,
                  **{name: value[name] for name in ("step", "sample", "observation", "state")},
                  "objective": list(objective), "reward": list(reward)}
        return json.dumps(answer) + "\n"


def request_core(request, *, binding=None):
    samples = {item.sample: (item.behavior_bits, item.reference_bits, item.advantage_bits) for item in request.samples}
    return Core(samples, request.steps, scalar.Profile(epsilon=request.epsilon, penalty=request.penalty), binding=binding,
                reference_source=request.reference_source)


def receiver(output, core):
    seen = len(output.getvalue().splitlines())

    def receive():
        nonlocal seen
        lines = output.getvalue().splitlines()
        answer = None
        for line in lines[seen:]:
            answer = core.reply(line) or answer
        seen = len(lines)
        if answer is None:
            raise ValueError("No learner step awaits cotangents")
        return answer

    return receive
