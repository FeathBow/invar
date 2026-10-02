import hashlib
import json
import struct
from collections.abc import Callable
from dataclasses import dataclass

from worker.cohort import unique

WORD_LIMIT = 1 << 32


def observation(words):
    return hashlib.sha256(struct.pack(f"<{len(words)}I", *words)).hexdigest()


def cotangent_words(value, count):
    if not isinstance(value, list) or len(value) != count or any(type(item) is not int or not 0 <= item < WORD_LIMIT for item in value):
        raise ValueError("Cotangents must be one FP32 word per reported token")
    return tuple(value)


@dataclass(frozen=True, kw_only=True)
class Exchange:
    binding: dict
    emit: Callable[[str, dict], None]
    receive: Callable[[], str]

    def proximal(self, *, sample, words):
        self.emit("proximal", {"binding": self.binding, "sample": sample, "words": list(words)})

    def reference(self, *, sample, words):
        self.emit("reference", {"binding": self.binding, "sample": sample, "words": list(words)})

    def current(self, *, step, sample, words, state):
        reported = {"binding": self.binding, "step": step, "sample": sample,
                    "observation": observation(words), "state": state}
        self.emit("current", {**reported, "words": list(words)})
        reply = json.loads(self.receive(), object_pairs_hook=unique)
        if not isinstance(reply, dict) or set(reply) != {"stage", *reported, "objective", "reward"}:
            raise ValueError("Expected the core's cotangents for the reported step")
        if reply["stage"] != "cotangents" or type(reply["step"]) is not int or any(reply[name] != value for name, value in reported.items()):
            raise ValueError("Cotangents answer a different learner step")
        return cotangent_words(reply["objective"], len(words)), cotangent_words(reply["reward"], len(words))

    def applied(self, *, step, before, after, consumed):
        self.emit("applied", {"binding": self.binding, "step": step, "before": before, "after": after,
                              "consumed": list(consumed)})
