from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Violation:
    left: np.ndarray
    right: np.ndarray
    symmetric: bool = False


@dataclass
class ViolationSet:
    cluster_indices: list[np.ndarray]
    row_to_cluster: np.ndarray | None = None
    violations: list[Violation] = field(default_factory=list)

    @property
    def num_clusters(self) -> int:
        return len(self.cluster_indices)

    @property
    def cluster_sizes(self) -> np.ndarray:
        return np.fromiter((len(rows) for rows in self.cluster_indices), dtype=int)

    @property
    def row_count(self) -> int:
        if self.row_to_cluster is not None:
            return len(self.row_to_cluster)
        return int(self.cluster_sizes.sum())

    def __len__(self) -> int:
        return sum(self._count_edges(violation) for violation in self.violations)

    def __bool__(self) -> bool:
        return len(self) > 0

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {"idx1": idx1, "idx2": idx2}
            for idx1, idx2 in self.iter_edges()
        ]
        return pd.DataFrame(rows, columns=["idx1", "idx2"])

    def iter_edges(self):
        for violation in self.violations:
            yield from self._iter_edges(violation)

    def _count_edges(self, violation: Violation) -> int:
        if violation.symmetric:
            n = self._cluster_member_count(violation.left)
            return n * (n - 1) // 2
        return self._cluster_member_count(violation.left) * self._cluster_member_count(violation.right)

    def _iter_edges(self, violation: Violation):
        left = self._rows_for_clusters(violation.left)
        right = left if violation.symmetric else self._rows_for_clusters(violation.right)

        if violation.symmetric:
            for i, idx1 in enumerate(left):
                for idx2 in left[i + 1 :]:
                    yield int(idx1), int(idx2)
            return

        for idx1 in left:
            for idx2 in right:
                if idx1 != idx2:
                    yield int(idx1), int(idx2)

    def _cluster_member_count(self, cluster_ids: np.ndarray) -> int:
        return int(sum(len(self.cluster_indices[int(cid)]) for cid in cluster_ids))

    def _rows_for_clusters(self, cluster_ids: np.ndarray) -> np.ndarray:
        if len(cluster_ids) == 0 or not self.cluster_indices:
            return np.array([], dtype=int)
        if len(cluster_ids) == 1:
            return self.cluster_indices[int(cluster_ids[0])]
        return np.concatenate([self.cluster_indices[int(cid)] for cid in cluster_ids])


@dataclass(frozen=True)
class CompactData:
    df: pd.DataFrame
    _compact_to_dense: list[np.ndarray]
    _dense_to_compact: np.ndarray
    attributes: list[str]

    def to_violation_set(self) -> ViolationSet:
        return ViolationSet(
            cluster_indices=self._compact_to_dense,
            row_to_cluster=self._dense_to_compact,
        )


def compact_data(data: pd.DataFrame, attributes: list[str]) -> CompactData:
    attributes = list(attributes)
    if data.empty:
        return CompactData(
            df=pd.DataFrame(columns=attributes),
            _compact_to_dense=[],
            _dense_to_compact=np.array([], dtype=int),
            attributes=attributes,
        )

    if not attributes:
        return CompactData(
            df=pd.DataFrame(index=[0]),
            _compact_to_dense=[np.arange(len(data), dtype=int)],
            _dense_to_compact=np.zeros(len(data), dtype=int),
            attributes=[],
        )

    grouped = data.groupby(attributes, dropna=False, sort=False).indices
    compact_to_dense = [np.asarray(indices, dtype=int) for indices in grouped.values()]
    dense_to_compact = np.zeros(len(data), dtype=int)
    for cluster_id, dense_indices in enumerate(compact_to_dense):
        dense_to_compact[dense_indices] = cluster_id

    first_rows = [int(indices[0]) for indices in compact_to_dense]
    compact_frame = data.iloc[first_rows][attributes].reset_index(drop=True)
    return CompactData(
        df=compact_frame,
        _compact_to_dense=compact_to_dense,
        _dense_to_compact=dense_to_compact,
        attributes=attributes,
    )


def violation_set_for(compact: CompactData) -> ViolationSet:
    return ViolationSet(
        cluster_indices=compact._compact_to_dense,
        row_to_cluster=compact._dense_to_compact,
    )
