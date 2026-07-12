import numpy as np
import pandas as pd

from marginals.marginals import Marginals

MIN_TARGET = 1e-5


class MarginalsErrorCalculator:
    def __init__(self, data: pd.DataFrame, marginals: Marginals):
        masks = [m.mask(data) for m in marginals]
        self.counts = np.array([mask.sum() for mask in masks], dtype=float)
        self.targets = np.array([m.target for m in marginals], dtype=float)
        self.safe_targets = np.maximum(self.targets, MIN_TARGET)
        self.matching = np.column_stack(masks) if masks else np.zeros((len(data), 0), dtype=bool)
        self.size = len(data)

    @property
    def has_marginals(self) -> bool:
        return len(self.targets) > 0

    def error(self, indices: np.ndarray | None = None) -> np.ndarray:
        return self._error_after_removal(indices) if indices is not None else self._current_error()

    def error_after_removing(self, indices: np.ndarray) -> np.ndarray:
        indices = np.atleast_1d(np.asarray(indices, dtype=int))
        if indices.size == 0:
            return self._current_error()

        counts_after_removal = self.counts - self.matching[indices].sum(axis=0)
        return self._error(counts_after_removal, self.size - indices.size)

    def _current_error(self) -> np.ndarray:
        return self._error(self.counts, self.size)

    def _error_after_removal(self, indices: np.ndarray) -> np.ndarray:
        indices = np.atleast_1d(np.asarray(indices, dtype=int))
        if not self.has_marginals:
            return np.zeros(indices.size)

        size_after_removal = self.size - 1
        if size_after_removal <= 0:
            return np.full(indices.size, np.inf)

        counts_after_removal = self.counts[None, :] - self.matching[indices, :]
        return (np.abs(self.targets - counts_after_removal / size_after_removal) / self.safe_targets).mean(axis=1)

    def _error(self, counts: np.ndarray, size: int) -> np.ndarray:
        return (np.abs(self.targets - counts / size) / self.safe_targets).mean()

    def remove(self, indices: np.ndarray) -> None:
        indices = np.atleast_1d(np.asarray(indices, dtype=int))
        if indices.size == 0:
            return
        if indices.size >= self.size:
            raise ValueError("Cannot remove all rows from the marginals error calculator.")

        self.counts -= self.matching[indices].sum(axis=0)
        self.size -= indices.size
