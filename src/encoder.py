from dataclasses import dataclass
from typing import Protocol, TypeAlias


class Encoder(Protocol):
    def encode(self, data: bytes): ...

    def decode(self, payload): ...


LZWCodeStream: TypeAlias = list[int]


@dataclass(frozen=True)
class LiteralToken:
    value: int


@dataclass(frozen=True)
class MatchToken:
    offset: int
    length: int


LZSSToken: TypeAlias = LiteralToken | MatchToken


class LZWEncoder:
    def encode(self, data: bytes) -> LZWCodeStream:
        if not data:
            return []

        dictionary = {bytes([i]): i for i in range(256)}
        next_code = 256
        codes: LZWCodeStream = []

        current = bytes([data[0]])
        for value in data[1:]:
            candidate = current + bytes([value])
            if candidate in dictionary:
                current = candidate
                continue

            codes.append(dictionary[current])
            dictionary[candidate] = next_code
            next_code += 1
            current = bytes([value])

        codes.append(dictionary[current])
        return codes

    def decode(self, codes: LZWCodeStream) -> bytes:
        if not codes:
            return b""

        dictionary = {i: bytes([i]) for i in range(256)}
        next_code = 256

        previous = dictionary[codes[0]]
        decoded = bytearray(previous)

        for code in codes[1:]:
            if code in dictionary:
                entry = dictionary[code]
            elif code == next_code:
                entry = previous + previous[:1]
            else:
                raise ValueError(f"Invalid LZW code: {code}")

            decoded.extend(entry)
            dictionary[next_code] = previous + entry[:1]
            next_code += 1
            previous = entry

        return bytes(decoded)


class LZSSEncoder:
    def __init__(
        self, window_size: int = 4096, lookahead_size: int = 18, min_match: int = 3
    ):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if lookahead_size <= 0:
            raise ValueError("lookahead_size must be positive")
        if min_match < 2:
            raise ValueError("min_match must be at least 2")

        self.window_size = window_size
        self.lookahead_size = lookahead_size
        self.min_match = min_match

    def encode(self, data: bytes) -> list[LZSSToken]:
        tokens: list[LZSSToken] = []
        index = 0

        while index < len(data):
            offset, length = self._find_longest_match(data, index)
            if length >= self.min_match:
                tokens.append(MatchToken(offset=offset, length=length))
                index += length
                continue

            tokens.append(LiteralToken(value=data[index]))
            index += 1

        return tokens

    def decode(self, tokens: list[LZSSToken]) -> bytes:
        decoded = bytearray()

        for token in tokens:
            if isinstance(token, LiteralToken):
                if not 0 <= token.value <= 255:
                    raise ValueError(f"Invalid literal value: {token.value}")
                decoded.append(token.value)
                continue

            if token.offset <= 0:
                raise ValueError(f"Invalid match offset: {token.offset}")
            if token.length <= 0:
                raise ValueError(f"Invalid match length: {token.length}")
            if token.offset > len(decoded):
                raise ValueError(
                    f"Match offset {token.offset} exceeds decoded size {len(decoded)}"
                )

            start = len(decoded) - token.offset
            for step in range(token.length):
                decoded.append(decoded[start + step])

        return bytes(decoded)

    def _find_longest_match(self, data: bytes, index: int) -> tuple[int, int]:
        window_start = max(0, index - self.window_size)
        best_offset = 0
        best_length = 0
        max_length = min(self.lookahead_size, len(data) - index)

        for candidate in range(window_start, index):
            length = 0
            while (
                length < max_length and data[candidate + length] == data[index + length]
            ):
                length += 1
                if candidate + length >= index:
                    # Allow overlap by reading from the bytes that would have just been emitted.
                    if data[index + length - 1] != data[candidate + length - 1]:
                        break

            if length > best_length:
                best_length = length
                best_offset = index - candidate

        return best_offset, best_length


__all__ = [
    "Encoder",
    "LZWCodeStream",
    "LiteralToken",
    "MatchToken",
    "LZSSToken",
    "LZWEncoder",
    "LZSSEncoder",
]
