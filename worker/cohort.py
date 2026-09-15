import math
import re
import struct
from collections import Counter
from dataclasses import dataclass

SPECIFICATION = "grpo-token-mean/v1"
WORD_BITS = 32


@dataclass(frozen=True, kw_only=True)
class Observation:
    sample: str
    group: str
    prompt: str
    seed: int
    limit: int
    temperature: float
    tokens: tuple[int, ...]
    prompt_length: int
    behavior_bits: tuple[int, ...]
    text: str
    truncated: bool
    reward: float
    advantage_bits: int


@dataclass(frozen=True, kw_only=True)
class Optimizer:
    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float


@dataclass(frozen=True, kw_only=True)
class BehaviorModel:
    base: str
    assembly: str


@dataclass(frozen=True, kw_only=True)
class Cohort:
    specification: str
    policy: str
    learner: str
    reference: str
    tokenizer: str
    base: str
    assembly: str
    behavior_model: BehaviorModel
    samples: tuple[Observation, ...]
    order: tuple[str, ...]
    epsilon: float
    penalty: float
    delta: float
    optimizer: Optimizer


def fields(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise ValueError(f"Expected exactly these request fields: {expected}")
    return value


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric request field")
    return float(value)


def identity(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA-256 identity")
    return value


def sample(value):
    value = fields(value, "sample group prompt seed limit temperature tokens prompt_length behavior_bits text truncated reward advantage_bits")
    if any(not isinstance(value[name], str) for name in ("sample", "group", "prompt", "text")):
        raise ValueError("Prompt and logical identities must be text")
    if not value["sample"] or not value["group"]:
        raise ValueError("Logical sample and group identities must be nonempty")
    if type(value["seed"]) is not int or type(value["limit"]) is not int or value["limit"] <= 0:
        raise ValueError("Sample seed must be integral and token limit positive")
    temperature = number(value["temperature"])
    if temperature <= 0:
        raise ValueError("Sample temperature must be positive")
    tokens, probabilities = observation(value)
    return Observation(**{**value, "temperature": temperature, "reward": number(value["reward"]),
                          "tokens": tokens, "behavior_bits": probabilities,
                          "advantage_bits": finite_word(value["advantage_bits"])})


def observation(value):
    tokens = token_ids(value["tokens"])
    words = behavior_words(value["behavior_bits"])
    prefix = value["prompt_length"]
    if type(prefix) is not int or not 0 < prefix < len(tokens):
        raise ValueError("Invalid observed prompt boundary")
    if len(words) != len(tokens) - prefix or len(words) > value["limit"]:
        raise ValueError("Behavior observations must match the admitted response tokens")
    if type(value["truncated"]) is not bool or (value["truncated"] and len(words) != value["limit"]):
        raise ValueError("Invalid observed truncation status")
    return tokens, words


def token_ids(values):
    if not isinstance(values, list) or any(type(item) is not int or item < 0 for item in values):
        raise ValueError("Observed token IDs must be nonnegative integers")
    return tuple(values)


def behavior_words(values):
    if not isinstance(values, list):
        raise ValueError("Behavior observations must be a sequence of FP32 words")
    return tuple(behavior_word(value) for value in values)


def finite_word(value):
    if type(value) is not int or not 0 <= value < 1 << WORD_BITS:
        raise ValueError("Expected an FP32 word")
    if not math.isfinite(struct.unpack("!f", struct.pack("!I", value))[0]):
        raise ValueError("Expected a finite FP32 word")
    return value


def behavior_word(value):
    finite_word(value)
    probability = struct.unpack("!f", struct.pack("!I", value))[0]
    if probability > 0:
        raise ValueError("Behavior log probabilities must be finite and nonpositive")
    return value


def optimizer(value):
    value = fields(value, "learning_rate betas epsilon weight_decay")
    if not isinstance(value["betas"], list) or len(value["betas"]) != 2:
        raise ValueError("AdamW requires two moment coefficients")
    result = Optimizer(learning_rate=number(value["learning_rate"]),
                       betas=tuple(number(item) for item in value["betas"]),
                       epsilon=number(value["epsilon"]), weight_decay=number(value["weight_decay"]))
    if result.learning_rate < 0 or result.epsilon <= 0 or result.weight_decay < 0:
        raise ValueError("Invalid AdamW scalar configuration")
    if any(not 0 <= beta < 1 for beta in result.betas):
        raise ValueError("AdamW moment coefficients must be in [0, 1)")
    return result


def behavior_model(value):
    fields(value, "base assembly")
    return BehaviorModel(base=identity(value["base"]), assembly=identity(value["assembly"]))


def logical_batch(samples, order):
    names = tuple(item.sample for item in samples)
    if not names or len(set(names)) != len(names):
        raise ValueError("Cohort samples must be nonempty and distinct")
    logical_order(order, names)
    groups = Counter(item.group for item in samples)
    if any(size < 2 for size in groups.values()):
        raise ValueError("Each advantage group requires at least two samples")


def logical_order(order, names):
    if not isinstance(order, list) or any(not isinstance(item, str) for item in order):
        raise ValueError("Logical update order must be a sequence of sample identities")
    if len(set(order)) != len(order) or set(order) != set(names):
        raise ValueError("Logical order must name every admitted sample exactly once")


def decode(value):
    value = fields(value, "specification policy learner reference tokenizer base assembly behavior_model samples order epsilon penalty delta optimizer")
    if value["specification"] != SPECIFICATION:
        raise ValueError("Unsupported update specification")
    if not isinstance(value["samples"], list):
        raise ValueError("Expected an explicit cohort sequence")
    samples = tuple(sample(item) for item in value["samples"])
    logical_batch(samples, value["order"])
    epsilon, penalty, delta = (number(value[name]) for name in ("epsilon", "penalty", "delta"))
    if not 0 < epsilon < 1 or penalty < 0 or delta <= 0:
        raise ValueError("Invalid GRPO coefficient configuration")
    return Cohort(specification=SPECIFICATION, policy=identity(value["policy"]),
                  learner=identity(value["learner"]), reference=identity(value["reference"]),
                  tokenizer=identity(value["tokenizer"]),
                  base=identity(value["base"]), assembly=identity(value["assembly"]),
                  behavior_model=behavior_model(value["behavior_model"]),
                  samples=samples, order=tuple(value["order"]), epsilon=epsilon,
                  penalty=penalty, delta=delta, optimizer=optimizer(value["optimizer"]))
