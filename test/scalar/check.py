import argparse
import hashlib
import json
import math
import platform
import random
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "worker"))
import scalar

SEED = 1729
ORDINARY_CASES = 512
RAW_CASES = 512
MEAN_CASES = 256
REFERENCE_TIMEOUT = 45
SIGN = 1 << 31
ROLES = ("current", "proximal", "behavior", "reference", "advantage")
WORD_WIDTH = 32
LENGTHS = (1, 2, 3, 4, 7, 16)
SPECIAL = (0, SIGN, 1, SIGN | 1, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000,
           0x3F800000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(encoded):
    return hashlib.sha256(encoded).hexdigest()


def word64(value):
    return struct.unpack("=Q", struct.pack("=d", value))[0]


def number64(value):
    return struct.unpack("=d", struct.pack("=Q", value))[0]


def token_case(values, *, epsilon=0.2, penalty=0.04, count=None):
    return {"kind": "tokens", "epsilon": word64(epsilon), "penalty": word64(penalty),
            "count": len(values) if count is None else count, "inputs": values}


def base():
    return {name: scalar.word(1 if name == "advantage" else -1) for name in ROLES}


def fixed():
    cases = [token_case([{**base(), role: value}]) for role in ROLES for value in SPECIAL]
    for epsilon in (0, 1, -1, math.nan, math.inf, 2 ** -1074, math.nextafter(1, 0), 0.5):
        cases.append(token_case([base()], epsilon=epsilon))
    for penalty in (-1, -0.0, 0, 2 ** -1074, 1e-40, 1e38, 1e100, math.nan, math.inf):
        cases.append(token_case([base()], penalty=penalty))
    for count in (0, -1, 2, (1 << 24) + 1):
        cases.append(token_case([base()], count=count))
    cases.extend((token_case([]), token_case([base(), base()], count=1)))
    for ratio in (0.25, 0.5, 1, 1.5, 2):
        difference = scalar.word(math.log(ratio))
        for offset in (-1, 0, 1):
            if not 0 <= difference + offset < (1 << WORD_WIDTH):
                continue
            value = scalar.number(difference + offset)
            for advantage in (-1, 0, 1, 2 ** -149):
                current, proximal = min(value, 0), min(-value, 0)
                values = {"current": scalar.word(current), "proximal": scalar.word(proximal),
                          "behavior": scalar.word(proximal), "reference": scalar.word(current),
                          "advantage": scalar.word(advantage)}
                cases.append(token_case([values], epsilon=0.5, penalty=0))
    for difference in (-104, -103.97208, -103, -90, -88, 88, 88.72284, 89):
        current, proximal = min(difference, 0), min(-difference, 0)
        cases.append(token_case([{**base(), "current": scalar.word(current),
                                  "proximal": scalar.word(proximal), "behavior": scalar.word(proximal)}]))
    for values in ([], [0.0], [-0.0], [2 ** 24, 1, -(2 ** 24)], [2 ** 24, -(2 ** 24), 1],
                   [1, 2 ** -24, -1], [3e38, 3e38, -3e38], [2 ** -149] * 3):
        cases.append({"kind": "mean", "values": [scalar.word(value) for value in values]})
    cases.extend({"kind": "mean", "values": [value]} for value in SPECIAL)
    return cases


def corpus():
    rng = random.Random(SEED)
    cases = fixed()
    for _ in range(ORDINARY_CASES):
        length = rng.choice(LENGTHS)
        inputs = [{**{name: scalar.word(rng.uniform(-20, 0)) for name in ROLES[:-1]},
                   "advantage": scalar.word(rng.uniform(-3, 3))} for _ in range(length)]
        cases.append(token_case(inputs, epsilon=rng.uniform(0.01, 0.99), penalty=rng.choice((0, 0.04, 0.125)),
                                count=length * rng.choice(LENGTHS)))
    for _ in range(RAW_CASES):
        inputs = [{name: rng.getrandbits(WORD_WIDTH) | (0 if name == "advantage" else SIGN)
                   for name in ROLES} for _ in range(rng.choice(LENGTHS))]
        cases.append(token_case(inputs))
    for _ in range(MEAN_CASES):
        values = [rng.getrandbits(WORD_WIDTH) for _ in range(rng.choice(LENGTHS))]
        cases.append({"kind": "mean", "values": values})
    return cases


def worker(case):
    try:
        if case["kind"] == "mean":
            result = scalar.mean32(case["values"])
        else:
            profile = scalar.Profile(epsilon=number64(case["epsilon"]), penalty=number64(case["penalty"]))
            values = tuple(scalar.Inputs(**value) for value in case["inputs"])
            result = scalar.document(scalar.calculate(profile, case["count"], values))
        return {"accepted": True, "words": result}
    except scalar.InvalidInput as error:
        return {"accepted": False, "kind": "invalid_input", "reason": str(error)}
    except scalar.NonFinite as error:
        return {"accepted": False, "kind": "non_finite", "reason": str(error)}


def equivalent(left, right):
    if left["accepted"] != right["accepted"]:
        return False
    return left["words"] == right["words"] if left["accepted"] else left["kind"] == right["kind"]


def run(options):
    if options.corpus is None:
        cases = corpus()
    else:
        previous = json.loads(options.corpus.read_bytes())
        cases = [case["input"] for case in previous["cases"]]
        if digest(canonical(cases)) != previous["inputs_sha256"]:
            raise ValueError("Frozen scalar corpus digest mismatch")
    encoded = canonical(cases)
    frozen = options.output.with_suffix(".inputs.json")
    with frozen.open("xb") as stream:
        stream.write(encoded)
    process = subprocess.run([str(options.reference)], input=encoded, capture_output=True, timeout=REFERENCE_TIMEOUT)
    if process.returncode:
        raise RuntimeError(f"Core scalar reference failed: {process.returncode}: {process.stderr.decode()}")
    core = json.loads(process.stdout)
    if len(core) != len(cases):
        raise ValueError("Core scalar reference omitted cases")
    observed = []
    for case, actual in zip(cases, core, strict=True):
        independent = worker(case)
        observed.append({"input": case, "core": actual, "worker": independent,
                         "equal": equivalent(actual, independent)})
    summary = {"cases": len(observed), "accepted": sum(value["core"]["accepted"] for value in observed),
               "rejected": sum(not value["core"]["accepted"] for value in observed),
               "differences": sum(not value["equal"] for value in observed)}
    result = {"format": "invar-scalar-corpus-v1", "reference": scalar.REFERENCE,
              "predicate": "exact accepted FP32 words and matching rejection class; messages retained separately",
              "inputs_sha256": digest(encoded), "reference_sha256": digest(options.reference.read_bytes()),
              "core_source_sha256": digest((ROOT / "src/Invar/Learn/Objective.hs").read_bytes()),
              "worker_source_sha256": digest((ROOT / "worker/scalar.py").read_bytes()),
              "checker_sha256": digest(Path(__file__).read_bytes()),
              "platform": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
                           "libc": platform.libc_ver()}, "summary": summary, "cases": observed}
    with options.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(json.dumps({**summary, "inputs_sha256": result["inputs_sha256"]}))
    return summary["differences"] == 0


def main():
    parser = argparse.ArgumentParser(description="Compare the independent core and worker scalar references exactly")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    raise SystemExit(0 if run(parser.parse_args()) else 1)


if __name__ == "__main__":
    main()
