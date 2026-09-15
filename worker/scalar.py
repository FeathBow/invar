import math
import struct
from dataclasses import dataclass

REFERENCE = "grpo-scalar-f32/v1"
WORD_LIMIT = 1 << 32
FIELDS = (("terms", "term"), ("current_gradient", "gradient"), ("reward_gradient", "reward_gradient"))


class InvalidInput(ValueError):
    pass


class NonFinite(ValueError):
    pass


@dataclass(frozen=True, kw_only=True)
class Profile:
    epsilon: float
    penalty: float

    def __post_init__(self):
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon < 1:
            raise InvalidInput("Clipping epsilon must be finite and between zero and one")
        if not math.isfinite(self.penalty) or self.penalty < 0:
            raise InvalidInput("Reference penalty must be finite and nonnegative")


@dataclass(frozen=True, kw_only=True)
class Inputs:
    current: int
    proximal: int
    behavior: int
    reference: int
    advantage: int


@dataclass(frozen=True, kw_only=True)
class Output:
    term: int
    gradient: int
    reward_gradient: int


@dataclass(frozen=True, kw_only=True)
class Constants:
    lower: float
    upper: float
    penalty: float
    count: float


def rounded(value):
    if not math.isfinite(value):
        raise NonFinite("Non-finite scalar arithmetic")
    try:
        result = struct.unpack("=f", struct.pack("=f", value))[0]
    except OverflowError as error:
        raise NonFinite("Scalar arithmetic overflows FP32") from error
    if not math.isfinite(result):
        raise NonFinite("Non-finite FP32 scalar result")
    return result


def word(value):
    return struct.unpack("=I", struct.pack("=f", rounded(value)))[0]


def number(encoded):
    if type(encoded) is not int or not 0 <= encoded < WORD_LIMIT:
        raise InvalidInput("Expected an unsigned FP32 word")
    value = struct.unpack("=f", struct.pack("=I", encoded))[0]
    if not math.isfinite(value):
        raise NonFinite("Non-finite FP32 scalar input")
    return value


def probability(encoded):
    value = number(encoded)
    if value > 0:
        raise InvalidInput("Log probabilities must be nonpositive")
    return value


def exponential(value):
    try:
        result = rounded(math.exp(value))
    except OverflowError as error:
        raise NonFinite("Probability ratio overflow") from error
    if result <= 0:
        raise InvalidInput("Probability ratio underflow")
    return result


def calculate(profile, total, inputs):
    if type(total) is not int or total <= 0 or not inputs or len(inputs) > total:
        raise InvalidInput("Invalid active token count")
    constants = Constants(lower=rounded(1 - profile.epsilon), upper=rounded(1 + profile.epsilon),
                          penalty=rounded(profile.penalty), count=rounded(total))
    return tuple(token(constants, value) for value in inputs)


def surrogate(constants, ratio, advantage):
    clipped = min(constants.upper, max(constants.lower, ratio))
    direct = rounded(ratio * advantage)
    bounded = rounded(clipped * advantage)
    if direct <= bounded:
        return direct, advantage
    slope = advantage if constants.lower < ratio < constants.upper else 0.0
    return bounded, slope


def token(constants, inputs):
    current, proximal, behavior, reference = map(probability, (
        inputs.current, inputs.proximal, inputs.behavior, inputs.reference))
    advantage = number(inputs.advantage)
    weight = exponential(rounded(proximal - behavior))
    ratio = exponential(rounded(current - proximal))
    difference = rounded(reference - current)
    reference_ratio = exponential(difference)
    selected, slope = surrogate(constants, ratio, advantage)
    reward = rounded(-weight * selected)
    distance = rounded(rounded(reference_ratio - difference) - 1)
    regularizer = rounded(constants.penalty * distance)
    value = rounded(reward + regularizer)
    current_slope = rounded(slope * ratio)
    reward_slope = rounded(-weight * current_slope)
    reference_slope = rounded(constants.penalty * rounded(1 - reference_ratio))
    objective_slope = rounded(reward_slope + reference_slope)
    return Output(term=word(value), gradient=word(rounded(objective_slope / constants.count)),
                  reward_gradient=word(rounded(reward_slope / constants.count)))


def mean32(encoded):
    if not encoded:
        raise InvalidInput("Empty objective mean")
    total = 0.0
    for value in encoded:
        total = rounded(total + number(value))
    return word(rounded(total / rounded(len(encoded))))


def document(outputs):
    return {name: [getattr(value, field) for value in outputs] for name, field in FIELDS}
