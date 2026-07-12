from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class ViolationLike(Protocol):
    left: np.ndarray
    right: np.ndarray
    symmetric: bool


class ViolationSetLike(Protocol):
    cluster_indices: list[np.ndarray]
    row_to_cluster: np.ndarray | None
    violations: Sequence[ViolationLike]
