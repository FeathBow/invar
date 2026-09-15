from dataclasses import asdict
import json

from worker.invocation import request as numerical_request
from worker import scalar
from worker.trajectory import Request


def requests(tasks):
    return tuple(Request(sample=item["name"], group=item["group"], prompt=item["prompt"],
                         seed=item["seed"], limit=item["tokens"], temperature=item["temperature"]) for item in tasks)


def observation(trajectory, identities):
    behavior = trajectory.behavior.tolist()
    return {"sample": trajectory.request.sample, **identities,
            "request": numerical_request(trajectory.request), "tokens": trajectory.tokens[0].tolist(),
            "prompt_length": trajectory.prompt_length, "behavior": behavior,
            "behavior_bits": [scalar.word(value) for value in behavior],
            "text": trajectory.text, "truncated": trajectory.truncated}


def identities(settings, *, behavior=False):
    prefix = "behavior-" if behavior else ""
    return {"adapter": settings["policy"], "tokenizer": settings["tokenizer-digest"],
            **{name: settings[prefix + name + "-digest"] for name in ("base", "assembly")}}


def probabilities(output, values):
    with (output / "numerical.json").open("x") as stream:
        json.dump({"format": "invar-native-objective-v1", "scalar_reference": scalar.REFERENCE,
                   "samples": [asdict(value) for value in values]}, stream, allow_nan=False)
        stream.write("\n")
