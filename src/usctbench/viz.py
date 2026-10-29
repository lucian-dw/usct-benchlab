"""Optional grayscale previews; numerical artifacts do not depend on Pillow."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_preview_png(image: np.ndarray, path: str | Path) -> Path:
    """Write a grayscale PNG preview for a 2-D image."""
    from PIL import Image

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

    Image.fromarray(scaled).save(out)
    return out
