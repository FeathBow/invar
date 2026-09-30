import math
import struct

WORD_LIMIT = 1 << 32


class InvalidInput(ValueError):
    pass


class NonFinite(ValueError):
    pass


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
