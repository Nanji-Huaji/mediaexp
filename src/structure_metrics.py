from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StructureMetrics:
    byte_entropy_bits: float
    normalized_byte_entropy: float
    window_entropy_bits: float
    normalized_window_entropy: float
    structure_score: float


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0

    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def byte_entropy(data: bytes) -> float:
    if not data:
        return 0.0

    values = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(values, minlength=256)
    return _entropy_from_counts(counts)


def window_entropy(data: bytes, window_size: int = 2) -> float:
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if len(data) < window_size:
        return 0.0

    if window_size == 1:
        return byte_entropy(data)

    values = np.frombuffer(data, dtype=np.uint8).astype(np.uint32)
    base = 256
    windows = values[: -window_size + 1].copy()
    for offset in range(1, window_size):
        windows = windows * base + values[offset : offset + len(windows)]

    _, counts = np.unique(windows, return_counts=True)
    return _entropy_from_counts(counts)


def normalized_entropy(entropy_bits: float, alphabet_size: int) -> float:
    if alphabet_size <= 1:
        return 0.0

    max_entropy = math.log2(alphabet_size)
    if max_entropy == 0:
        return 0.0
    return max(0.0, min(1.0, entropy_bits / max_entropy))


def analyze_structure(data: bytes, window_size: int = 2) -> StructureMetrics:
    byte_entropy_bits = byte_entropy(data)
    normalized_byte = normalized_entropy(byte_entropy_bits, 256)

    window_entropy_bits = window_entropy(data, window_size=window_size)
    normalized_window = normalized_entropy(window_entropy_bits, 256**window_size)

    # Lower entropy means stronger regularity, so use one minus the average entropy.
    structure_score = 1.0 - ((normalized_byte + normalized_window) / 2.0)
    return StructureMetrics(
        byte_entropy_bits=round(byte_entropy_bits, 4),
        normalized_byte_entropy=round(normalized_byte, 4),
        window_entropy_bits=round(window_entropy_bits, 4),
        normalized_window_entropy=round(normalized_window, 4),
        structure_score=round(structure_score, 4),
    )


__all__ = [
    "StructureMetrics",
    "analyze_structure",
    "byte_entropy",
    "normalized_entropy",
    "window_entropy",
]
