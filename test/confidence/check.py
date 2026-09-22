from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import json
import subprocess
import sys


def main():
    alphas = [Fraction(1, 20), Fraction(1, 40), Fraction(1, 10**40), Fraction(10**60 - 1, 10**60)]
    cases = [(bound, alpha, count) for bound in (Fraction(1), Fraction(2), Fraction(1, 2))
             for alpha in alphas for count in (1, 2, 32, 128, 600, 14979)]
    inputs = [f"{bound.numerator} {bound.denominator} {alpha.numerator} {alpha.denominator} {count}" for bound, alpha, count in cases]
    invalid = ["1 1 0 1 32", "1 1 1 1 32", "1 1 2 1 32", "1 1 1 20 0", "0 1 1 20 32", "-1 1 1 20 32", "1 1 1 20 -1"]
    result = subprocess.run([sys.argv[1]], input="\n".join(inputs + invalid) + "\n", text=True, capture_output=True, check=True)
    output = result.stdout.splitlines()
    assert len(output) == len(inputs) + len(invalid)
    records = []
    with localcontext() as context:
        context.prec = 120
        for (bound, alpha, count), measured in zip(cases, output[:len(inputs)], strict=True):
            numerator, denominator = map(int, measured.split())
            upper = Decimal(numerator) / Decimal(denominator)
            rate = Decimal(alpha.numerator) / Decimal(alpha.denominator)
            width = (Decimal(bound.numerator) / Decimal(bound.denominator)) * ((-rate.ln()) / (2 * count)).sqrt()
            assert upper >= width, (bound, alpha, count, upper, width)
            # Near alpha=1 the square root amplifies the uniform log enclosure.
            assert upper - width < Decimal("1e-12"), (bound, alpha, count, upper, width)
            records.append(dict(range=str(bound), alpha=str(alpha), count=count, upper=measured, decimal120=str(width)))
    assert output[len(inputs):] == ["invalid"] * len(invalid)
    record = dict(directional_checks=len(cases), invalid_cases=len(invalid), checks=records,
                  scope="finite independent numerical cross-check; sampling and selection remain external premises")
    Path(sys.argv[2]).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key:record[key] for key in ("directional_checks", "invalid_cases", "scope")}))


if __name__ == "__main__":
    main()
