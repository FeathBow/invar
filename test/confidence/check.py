from decimal import Decimal, localcontext
from fractions import Fraction
import random
import subprocess
import sys


def decimal(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def hoeffding(bound, alpha, count):
    return decimal(bound) * ((-decimal(alpha).ln()) / (2 * count)).sqrt()


def bernstein(bound, alpha, samples):
    count = len(samples)
    mean = sum(samples, Fraction()) / count
    variance = sum(((value - mean) ** 2 for value in samples), Fraction()) / (count - 1)
    logarithm = (2 / decimal(alpha)).ln()
    return (2 * decimal(variance) * logarithm / count).sqrt() + 7 * decimal(bound) * logarithm / (3 * (count - 1))


def line(values):
    return " ".join(str(value) for value in values)


def encoded(samples):
    return [part for value in samples for part in (value.numerator, value.denominator)]


def main():
    alphas = [Fraction(1, 20), Fraction(1, 40), Fraction(1, 10**40), Fraction(10**60 - 1, 10**60)]
    ranges = [Fraction(1), Fraction(2), Fraction(1, 2)]
    generator = random.Random(1701)
    cases, inputs = [], []
    for bound in ranges:
        for alpha in alphas:
            for count in (1, 2, 32, 128, 600, 14979):
                cases.append((hoeffding, bound, alpha, count))
                inputs.append(line([bound.numerator, bound.denominator, alpha.numerator, alpha.denominator, count]))
            for count in (2, 3, 32, 600):
                low = -bound / 2 if generator.random() < 0.5 else Fraction()
                samples = [low + Fraction(generator.randint(0, 64), 64) * bound for _ in range(count)]
                cases.append((bernstein, bound, alpha, samples))
                inputs.append(line(["bernstein", bound.numerator, bound.denominator, alpha.numerator, alpha.denominator, *encoded(samples)]))
    losses = [Fraction(1)] * 527 + [Fraction(0)] * 1473
    increases = [Fraction(1)] * 21 + [Fraction(-1)] * 20 + [Fraction(0)] * 1959
    for bound, samples in ((Fraction(1), losses), (Fraction(2), increases)):
        cases.append((bernstein, bound, Fraction(1, 40), samples))
        inputs.append(line(["bernstein", bound.numerator, bound.denominator, 1, 40, *encoded(samples)]))
    invalid = ["1 1 0 1 32", "1 1 1 1 32", "1 1 2 1 32", "1 1 1 20 0", "0 1 1 20 32", "-1 1 1 20 32", "1 1 1 20 -1",
               "bernstein 1 1 1 20 1 2", "bernstein 1 1 1 20", "bernstein 1 1 0 1 0 1 1 1", "bernstein 1 1 1 1 0 1 1 1",
               "bernstein 1 2 1 20 0 1 1 1", "bernstein 0 1 1 20 0 1 1 2", "bernstein 1 1 1 20 0 1 1"]
    result = subprocess.run([sys.argv[1]], input="\n".join(inputs + invalid) + "\n", text=True, capture_output=True, check=True)
    output = result.stdout.splitlines()
    assert len(output) == len(inputs) + len(invalid)
    with localcontext() as context:
        context.prec = 120
        for (reference, bound, alpha, argument), measured in zip(cases, output[:len(inputs)], strict=True):
            numerator, denominator = map(int, measured.split())
            upper = Decimal(numerator) / Decimal(denominator)
            width = reference(bound, alpha, argument)
            assert upper >= width, (reference.__name__, bound, alpha, upper, width)
            assert upper - width < Decimal("1e-12"), (reference.__name__, bound, alpha, upper, width)
    assert output[len(inputs):] == ["invalid"] * len(invalid), output[len(inputs):]
    print(len(cases), "bounds checked,", len(invalid), "invalid inputs refused")


if __name__ == "__main__":
    main()
