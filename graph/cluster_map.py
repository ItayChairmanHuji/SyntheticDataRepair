from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EMPTY_MEMBERS = np.array([], dtype=np.int64)


@dataclass(frozen=True, slots=True)
class ClusterMap:
    row_to_cluster: np.ndarray
    members: tuple[np.ndarray, ...]

    @classmethod
    def empty(cls, n: int) -> ClusterMap:
        return cls(np.full(n, -1, dtype=np.int64), ())

    @classmethod
    def from_parts(
        cls, n: int, row_to_cluster: np.ndarray | None, members: list[np.ndarray] | None
    ) -> ClusterMap:
        if row_to_cluster is None:
            if members:
                raise ValueError("members cannot be provided without row_to_cluster")
            return cls.empty(n)
        if members is None:
            raise ValueError("members are required when row_to_cluster is provided")

        row_to_cluster = np.asarray(row_to_cluster, dtype=np.int64)
        if len(row_to_cluster) != n:
            raise ValueError(f"row_to_cluster has length {len(row_to_cluster)}, expected {n}")

        normalized_members = tuple(np.asarray(rows, dtype=np.int64) for rows in members)
        return cls(row_to_cluster, normalized_members)

    @property
    def num_clusters(self) -> int:
        return len(self.members)

    def cluster_of(self, row_idx: int) -> int:
        return int(self.row_to_cluster[row_idx])

    def members_for(self, cluster_ids: np.ndarray) -> np.ndarray:
        if len(cluster_ids) == 0:
            return EMPTY_MEMBERS
        if len(cluster_ids) == 1:
            return self.members[int(cluster_ids[0])]
        return np.concatenate([self.members[int(cid)] for cid in cluster_ids])

    def count_members(self, cluster_ids: np.ndarray) -> int:
        return int(sum(len(self.members[int(cid)]) for cid in cluster_ids))
