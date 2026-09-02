from __future__ import annotations

import numpy as np


def image(height: int, width: int) -> np.ndarray:
    values = np.arange(height * width * 3, dtype=np.uint64)
    return ((values * 73 + values // 7 * 19) & 255).astype(np.uint8).reshape(height, width, 3)


def as_array(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()  # type: ignore[union-attr, no-any-return]
    return np.asarray(value)
