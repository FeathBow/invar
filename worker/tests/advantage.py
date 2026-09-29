import math
import struct
from dataclasses import dataclass



@dataclass(frozen=True, kw_only=True)
class Reward:
    sample: str
    group: str
    value: float


def advantages(rewards, delta):
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("Advantage delta must be finite and positive")
    if not rewards or len({item.sample for item in rewards}) != len(rewards):
        raise ValueError("Rewards must contain distinct logical sample identities")
    groups = {}
    for item in sorted(rewards, key=lambda reward: reward.sample):
        if not item.sample or not item.group or not math.isfinite(item.value):
            raise ValueError("Reward identity and value must be defined")
        groups.setdefault(item.group, []).append(item)
    result = []
    for group in groups.values():
        if len(group) < 2:
            raise ValueError("Each logical reward group requires at least two samples")
        mean = math.fsum(item.value for item in group) / len(group)
        centered = tuple(item.value - mean for item in group)
        if any(not math.isfinite(value) for value in centered):
            raise ValueError("Non-finite centered reward")
        variance = math.fsum(value * value for value in centered) / len(group)
        if not math.isfinite(variance):
            raise ValueError("Non-finite reward variance")
        denominator = math.sqrt(variance) + delta
        if not math.isfinite(denominator):
            raise ValueError("Non-finite advantage denominator")
        result.extend((item.sample, value / denominator) for item, value in zip(group, centered, strict=True))
    if any(not math.isfinite(value) for _, value in result):
        raise ValueError("Non-finite group advantage")
    return tuple(sorted(result))


@dataclass(frozen=True, kw_only=True)
class Checked:
    rewards: tuple[Reward, ...]
    values: tuple[tuple[str, float], ...]


def word(value):
    encoded = struct.pack("!f", value)
    if not math.isfinite(struct.unpack("!f", encoded)[0]):
        raise ValueError("Advantage must be finite in FP32")
    return struct.unpack("!I", encoded)[0]


def check(request):
    rewards = tuple(Reward(sample=item.sample, group=item.group, value=item.reward)
                    for item in request.samples)
    values = advantages(rewards, request.delta)
    actual = {sample: word(value) for sample, value in values}
    for item in request.samples:
        if actual[item.sample] != item.advantage_bits:
            raise ValueError(f"Advantage differs from the core expectation for sample {item.sample!r}")
    return Checked(rewards=rewards, values=values)
