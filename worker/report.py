from worker import float32

from worker.invocation import request as reported_request
from worker import registry

def ready(call, identities, *, model, previous, emit):
    if previous is not None:
        emit("unloaded_adapter", registry.invocation(previous))
    materialization = {name: identities[name] for name in ("tokenizer", "base", "assembly")}
    emit("loaded_adapter", {"binding": call.invocation.binding(), "requested": call.identities["adapter"],
                            "consumed": identities["adapter"], **materialization,
                            "load": registry.invocation(call.load), "image": registry.image(identities),
                            "model": model[0], "revision": model[1],
                            "scope": "tensor binding; not publication or numerical certification"})
    emit("consumed", {"binding": call.invocation.binding(), "program": call.invocation.program,
                      "load": registry.invocation(call.load), **identities,
                      "request": reported_request(call.request)})


def result(call, trajectory, *, identities, emit, reference=None):
    emit("result", {"binding": call.invocation.binding(), **identities,
                    "request": reported_request(trajectory.request),
                    "tokens": trajectory.tokens[0].tolist(), "prompt_length": trajectory.prompt_length,
                    "behavior": trajectory.behavior.tolist(), "text": trajectory.text,
                    "behavior_bits": [float32.word(value) for value in trajectory.behavior.tolist()],
                    "truncated": trajectory.truncated,
                    "reference": None if reference is None else {"adapter": reference[0], "bits": list(reference[1])}})
