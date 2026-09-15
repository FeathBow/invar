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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker"))

from advantage import word
import implementation
from objective import Reward, advantages

SEED = 1729
RANDOM_CASES = 512
MAX_MEMBERS = 32
WORD_BITS = 64
TIMEOUT_SECONDS = 60
DELTA = 1e-4


def bits(value):
    return struct.unpack("!Q", struct.pack("!d", value))[0]


def value(encoded):
    return struct.unpack("!d", struct.pack("!Q", encoded))[0]


def grouped(values, *, groups=None, delta=DELTA):
    assigned = ["g0"] * len(values) if groups is None else groups
    return {"kind": "advantage", "delta": bits(delta),
            "rewards": [{"sample": f"s{index}", "group": group, "bits": bits(reward)}
                        for index, (group, reward) in enumerate(zip(assigned, values, strict=True))]}


def corpus():
    fixed = [("binary", [0, 1]), ("zero", [0, 0]), ("one", [1, 1]),
             ("signed-zero", [-0.0, 0.0]), ("decimal", [0.1] * 3),
             ("cancellation", [1e16, 1, -1e16]),
             ("halfway-down", [1, 2 ** -53]), ("halfway-up", [1, 2 ** -53, 2 ** -54]),
             ("subnormal", [math.ulp(0.0), -math.ulp(0.0)]),
             ("intermediate-overflow", [1e308, 1e308, -1e308]),
             ("square-overflow", [-1e200, 1e200]), ("empty", []),
             ("singleton", [1]), ("infinity", [float("inf"), 0]), ("nan", [float("nan"), 0])]
    cases = [{"name": f"{kind}-{name}", "input": grouped(values) if kind == "advantage"
              else {"kind": "sum", "values": [bits(item) for item in values]}}
             for name, values in fixed for kind in ("sum", "advantage")]
    for delta in (0.0, -1.0, math.ulp(0.0), sys.float_info.max, float("inf"), float("nan")):
        cases.append({"name": f"delta-{bits(delta)}", "input": grouped([0, 1], delta=delta)})
    rng = random.Random(SEED)
    for index in range(RANDOM_CASES):
        count = rng.randint(2, MAX_MEMBERS)
        raw = [value(rng.getrandbits(WORD_BITS)) for _ in range(count)]
        ordinary = [rng.uniform(-100, 100) for _ in range(count)]
        for category, values in (("raw", raw), ("ordinary", ordinary)):
            cases.append({"name": f"sum-{category}-{index}",
                          "input": {"kind": "sum", "values": [bits(item) for item in values]}})
            original = grouped(values)
            cases.append({"name": f"advantage-{category}-{index}", "input": original})
            reordered = {**original, "rewards": rng.sample(original["rewards"], len(values))}
            cases.append({"name": f"permutation-{category}-{index}", "input": reordered})
    cases.extend([{"name": "grouping-original", "input": grouped([0, 0, 1, 1], groups=["a", "a", "b", "b"])},
                  {"name": "grouping-mutant", "input": grouped([0, 0, 1, 1], groups=["a", "b", "a", "b"])}])
    return cases


def observe(case):
    try:
        if case["kind"] == "sum":
            actual = math.fsum(value(item) for item in case["values"])
            if not math.isfinite(actual):
                raise ValueError("Non-finite exact sum")
            words = bits(actual)
        else:
            rewards = tuple(Reward(sample=item["sample"], group=item["group"], value=value(item["bits"]))
                            for item in case["rewards"])
            words = {sample: word(amount) for sample, amount in advantages(rewards, value(case["delta"]))}
        return {"accepted": True, "words": words}
    except (ValueError, OverflowError) as problem:
        return {"accepted": False, "reason": f"{type(problem).__name__}: {problem}"}


def compare(cases, reference):
    payload = json.dumps([case["input"] for case in cases], sort_keys=True, separators=(",", ":")).encode()
    process = subprocess.run([str(reference)], input=payload, capture_output=True, check=True, timeout=TIMEOUT_SECONDS)
    core = json.loads(process.stdout)
    if not isinstance(core, list) or len(core) != len(cases):
        raise ValueError("Core reference returned a different case inventory")
    rows = []
    for case, expected in zip(cases, core, strict=True):
        actual = observe(case["input"])
        equal = actual["accepted"] == expected["accepted"]
        equal &= not actual["accepted"] or actual.get("words") == expected.get("words")
        rows.append({**case, "core": expected, "worker": actual, "equal": equal})
    return payload, rows


def main():
    parser = argparse.ArgumentParser(description="Compare the independent Haskell and CPython advantage references")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, help="Replay the exact inputs of an earlier comparison artifact")
    options = parser.parse_args()
    frozen = None if options.corpus is None else json.loads(options.corpus.read_bytes())
    cases = corpus() if frozen is None else [{"name": row["name"], "input": row["input"]} for row in frozen["cases"]]
    payload, rows = compare(cases, options.reference)
    corpus_digest = hashlib.sha256(payload).hexdigest()
    if frozen is not None and corpus_digest != frozen["summary"]["corpus_sha256"]:
        raise ValueError("Replay inputs differ from the frozen corpus digest")
    original, mutant = rows[-2:]
    effective = all(item["core"]["accepted"] and item["worker"]["accepted"] for item in (original, mutant))
    effective &= original["core"].get("words") != mutant["core"].get("words")
    summary = {"seed": SEED, "cases": len(rows), "mismatches": sum(not row["equal"] for row in rows),
               "accepted": sum(row["core"]["accepted"] for row in rows),
               "rejected": sum(not row["core"]["accepted"] for row in rows), "grouping_effective": effective,
               "corpus_sha256": corpus_digest,
               "reference_sha256": hashlib.sha256(options.reference.read_bytes()).hexdigest(),
               "comparator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "implementation_sha256": hashlib.sha256(json.dumps(implementation.current(), sort_keys=True,
                                                                   separators=(",", ":")).encode()).hexdigest(),
               "python": sys.version, "platform": platform.platform(), "machine": platform.machine()}
    document = {"summary": summary, "cases": rows}
    with options.output.open("x") as output:
        json.dump(document, output, sort_keys=True, separators=(",", ":"), allow_nan=False)
        output.write("\n")
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(summary["mismatches"] != 0 or not effective)


if __name__ == "__main__":
    main()
