from fractions import Fraction
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from worker.tests.probe import decimal_kl as direct

def word(value):
    return struct.unpack("=I", struct.pack("=f", value))[0]


def cases():
    fixed = [([word(0.5), word(0.5)], [word(0.25), word(0.75)]),
             ([word(0.5), word(0.5)], [word(0.25), word(0.25)]),
             ([word(1), 0], [word(0.5), word(0.5)]),
             ([word(1), 0], [0, word(1)]),
             ([word(1), 1], [1, word(1)]),
             ([0x3f000001, 0x3effffff], [0x3f000000, 0x3f000000]),
             ([1, 2, 3], [2, 4, 6]),
             ([word(1), 0x80000000], [word(1), 0]),
             ([word(1), 1], [word(1), 2])]
    generator = random.Random(1701)
    random_cases = []
    for _ in range(256):
        size = generator.randint(1, 32)
        vectors = [[(generator.randrange(127) << 23) | generator.getrandbits(23) for _ in range(size)] for _ in range(2)]
        random_cases.append(tuple(vectors))
    return fixed + random_cases


def check_bounds(actual, expected):
    if expected.is_infinite():
        assert actual == {"kind": "positive_infinity"}, actual
        return None
    assert actual["kind"] == "finite", actual
    lower, upper = (Fraction(*actual[key]) for key in ("lower", "upper"))
    assert lower <= Fraction(expected) <= upper, (actual, str(expected))
    return upper - lower


def invalid_cases():
    initial = [([], []), ([0], [0]), ([word(1)], [word(1), 0]),
               ([0x7fc00000], [word(1)]), ([0x7f800000], [word(1)]),
               ([0xbf800000], [word(1)]), ([word(2)], [word(1)])]
    tail_length = 2048
    prefix = [word(0.5)] * tail_length
    for bad in (0x7fc00000, 0x7f800000, 0xbf800000, word(2)):
        initial.extend((([word(1), bad], [0, word(1)]),
                        ([0, word(1)], [word(1), bad]),
                        (prefix + [bad], prefix + [word(0.5)]),
                        (prefix + [word(0.5)], prefix + [bad])))
    return initial


def main():
    binary, output = Path(sys.argv[1]), Path(sys.argv[2])
    vectors = cases()
    invalid = invalid_cases()
    block = vectors[-1]
    repeats = 4096
    large = tuple(values * repeats for values in block)
    supplied = vectors + invalid + [large]
    encoded = "\n".join("(" + str(a) + "," + str(b) + ")" for a, b in supplied) + "\n"
    started = time.monotonic()
    result = subprocess.run([str(binary)], input=encoded, capture_output=True, text=True, timeout=55)
    elapsed = time.monotonic() - started
    assert result.returncode == 0, result.stderr
    decoded = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(decoded) == len(supplied)
    checked = []
    for index, ((left, right), actual) in enumerate(zip(vectors, decoded[:len(vectors)], strict=True)):
        for direction, (p, q), bound in zip(("forward", "backward"), ((left, right), (right, left)), actual, strict=True):
            expected = direct(p, q)
            width = check_bounds(bound, expected)
            checked.append(dict(case=index, direction=direction, bound=bound, decimal100=str(expected),
                                width=None if width is None else str(width)))
    for actual in decoded[len(vectors):len(vectors)+len(invalid)]:
        assert "error" in actual, actual
    for (p, q), actual in zip((block, block[::-1]), decoded[-1], strict=True):
        check_bounds(actual, direct(p, q))
    record = dict(finite_cases=len(vectors), directions_checked=len(checked), invalid_cases=len(invalid),
                  repeated_block_vocabulary=len(large[0]), process_seconds=elapsed, checks=checked,
                  meaning="exact normalization of represented FP32 masses; not model or sampler correctness")
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "checks"}))


if __name__ == "__main__":
    main()
