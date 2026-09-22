import codecs
from dataclasses import dataclass
import hashlib
import json
import re

from worker import core

CHUNK_BYTES = 64 * 1024
WHITESPACE = " \t\r\n"
STRUCTURE = re.compile(r'[\[\]{}"]')
STRING = re.compile(r'["\\]')
DELIMITER = re.compile(r'[ \t\r\n,\]}]')
BYTE_MARKS = ((codecs.BOM_UTF32_LE, "utf-32-le"), (codecs.BOM_UTF32_BE, "utf-32-be"),
              (codecs.BOM_UTF8, "utf-8"), (codecs.BOM_UTF16_LE, "utf-16-le"), (codecs.BOM_UTF16_BE, "utf-16-be"))


@dataclass(frozen=True, kw_only=True)
class Location:
    offset: int
    size: int
    sha256: str


class Reader:
    def __init__(self, stream):
        self.stream = stream
        prefix = stream.read(4)
        mark, self.encoding = next(((mark, encoding) for mark, encoding in BYTE_MARKS if prefix.startswith(mark)),
                                   (b"", json.detect_encoding(prefix)))
        self.decoder = codecs.getincrementaldecoder(self.encoding)(errors="surrogatepass")
        self.buffer = self.decoder.decode(prefix[len(mark):])
        self.position = len(mark)
        self.byte_count = len(prefix)
        self.digest = hashlib.sha256(prefix)
        self.ended = False

    def fill(self):
        while not self.buffer and not self.ended:
            encoded = self.stream.read(CHUNK_BYTES)
            self.byte_count += len(encoded)
            self.digest.update(encoded)
            self.ended = not encoded
            self.buffer = self.decoder.decode(encoded, final=self.ended)

    def peek(self):
        self.fill()
        return self.buffer[:1]

    def take(self, count):
        self.fill()
        text = self.buffer[:count]
        self.buffer = self.buffer[count:]
        self.position += len(text.encode(self.encoding, errors="surrogatepass"))
        return text

    def space(self):
        while self.peek() and self.buffer[0] in WHITESPACE:
            self.take(len(self.buffer) - len(self.buffer.lstrip(WHITESPACE)))

    def expect(self, character):
        self.space()
        if self.take(1) != character:
            raise ValueError("Expected JSON delimiter: " + character)

    def until(self, pattern):
        pieces = []
        while self.peek():
            found = pattern.search(self.buffer)
            pieces.append(self.take(len(self.buffer) if found is None else found.start()))
            if found is not None:
                break
        return "".join(pieces)

    def quoted(self):
        pieces = [self.take(1)]
        while True:
            pieces.append(self.until(STRING))
            character = self.take(1)
            if not character:
                raise ValueError("Incomplete JSON string")
            pieces.append(character)
            if character == '"':
                return "".join(pieces)
            escaped = self.take(1)
            if not escaped:
                raise ValueError("Incomplete JSON escape")
            pieces.append(escaped)

    def compound(self):
        pieces, depth = [], 0
        while True:
            pieces.append(self.until(STRUCTURE))
            character = self.peek()
            if not character:
                raise ValueError("Incomplete JSON object or array")
            if character == '"':
                pieces.append(self.quoted())
                continue
            pieces.append(self.take(1))
            depth += 1 if character in "[{" else -1
            if depth == 0:
                return "".join(pieces)

    def value(self):
        self.space()
        offset, first = self.position, self.peek()
        if not first:
            raise ValueError("Missing JSON value")
        if first in "[{":
            text = self.compound()
        elif first == '"':
            text = self.quoted()
        else:
            text = self.until(DELIMITER)
        value = core.decode(text)
        encoded = text.encode(self.encoding, errors="surrogatepass")
        return value, Location(offset=offset, size=len(encoded), sha256=hashlib.sha256(encoded).hexdigest())

    def items(self, opening, closing):
        self.expect(opening)
        self.space()
        if self.peek() == closing:
            self.take(1)
            return
        while True:
            yield
            self.space()
            if self.peek() == closing:
                self.take(1)
                return
            self.expect(",")

    def members(self):
        seen = set()
        for _ in self.items("{", "}"):
            key, _ = self.value()
            if not isinstance(key, str) or key in seen:
                raise ValueError("Expected unique JSON object keys")
            seen.add(key)
            self.expect(":")
            yield key

    def finish(self):
        self.space()
        if self.peek():
            raise ValueError("Unexpected data after the probe JSON object")
