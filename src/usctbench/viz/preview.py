"""Tiny PNG preview writer with no plotting dependency."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def write_preview_png(image: np.ndarray, path: str | Path) -> Path:
    """Write a grayscale PNG preview for a 2-D image."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image, dtype=float)
    if array.ndim != 2:
        raise ValueError("preview image must be 2-D")
    finite = np.isfinite(array)
    if not np.any(finite):
        scaled = np.zeros(array.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(array[finite], [1.0, 99.0])
        if high <= low:
            high = low + 1.0
        scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
        scaled = np.where(finite, scaled, 0.0)
        scaled = (255.0 * scaled).astype(np.uint8)

    height, width = scaled.shape
    raw = b"".join(b"\x00" + scaled[row].tobytes() for row in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    png += _chunk(b"IDAT", zlib.compress(raw))
    png += _chunk(b"IEND", b"")
    out.write_bytes(png)
    return out


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

