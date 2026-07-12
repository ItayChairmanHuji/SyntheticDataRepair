from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .block_pair import BlockSide
from .cluster_map import ClusterMap
from .types import ViolationLike, ViolationSetLike


@dataclass(frozen=True, slots=True)
class ClusterBlockMembership:
    block_pair_idx: int
    side: BlockSide


@dataclass(frozen=True, slots=True)
class ClusterBlockPair:
    left_clusters: np.ndarray
    right_clusters: np.ndarray
    union_clusters: np.ndarray
    is_clique: bool = False

    @classmethod
    def from_conflict(cls, conflict: ViolationLike) -> ClusterBlockPair:
        left_clusters = np.asarray(conflict.left, dtype=np.int64)
        right_clusters = left_clusters if conflict.symmetric else np.asarray(conflict.right, dtype=np.int64)
        union_clusters = left_clusters if conflict.symmetric else _combined_clusters(left_clusters, right_clusters)
        return cls(
            left_clusters=left_clusters,
            right_clusters=right_clusters,
            union_clusters=union_clusters,
            is_clique=conflict.symmetric,
        )


class ClusterCountGraph:
    """Conflict graph over compact-data clusters.

    Rows in the same compact cluster have identical conflict neighborhoods. This
    representation keeps one degree and active count per cluster, then emits a
    concrete row index only when a row from that cluster is selected.
    """

    def __init__(self, n: int, violation_set: ViolationSetLike) -> None:
        self.n = n
        self.cluster_map = ClusterMap.from_parts(n, violation_set.row_to_cluster, violation_set.cluster_indices)
        self.cluster_sizes = np.asarray([len(rows) for rows in self.cluster_map.members], dtype=np.int64)
        self.remaining = self.cluster_sizes.copy()
        self.removed = np.zeros(len(self.cluster_sizes), dtype=np.int64)
        self.degrees = np.zeros(len(self.cluster_sizes), dtype=np.int64)
        self.block_pairs = [ClusterBlockPair.from_conflict(conflict) for conflict in violation_set.violations]
        self.cluster_to_blocks: list[list[ClusterBlockMembership]] = [
            [] for _ in range(self.cluster_map.num_clusters)
        ]
        self._build()

    def has_edges(self) -> bool:
        return bool(np.any((self.remaining > 0) & (self.degrees > 0)))

    def pop_max_degree_vertex(self) -> int:
        if not self.has_edges():
            raise ValueError("No edges left in graph")

        active_degrees = np.where(self.remaining > 0, self.degrees, -1)
        cluster_id = int(active_degrees.argmax())
        if self.degrees[cluster_id] <= 0:
            raise ValueError("No edges left in graph")

        row_idx = self._pop_row(cluster_id)
        self._update_neighbor_clusters(cluster_id)
        return row_idx

    def _build(self) -> None:
        for block_idx, block_pair in enumerate(self.block_pairs):
            self._add_block_pair(block_idx, block_pair)

    def _add_block_pair(self, block_idx: int, block_pair: ClusterBlockPair) -> None:
        left_cids = {int(cid) for cid in block_pair.left_clusters}
        right_cids = {int(cid) for cid in block_pair.right_clusters}
        left_count = _cluster_count(self.cluster_sizes, block_pair.left_clusters)
        right_count = left_count if block_pair.is_clique else _cluster_count(self.cluster_sizes, block_pair.right_clusters)
        union_count = left_count if block_pair.is_clique else _cluster_count(self.cluster_sizes, block_pair.union_clusters)

        for cid in left_cids | right_cids:
            if self.cluster_sizes[cid] == 0:
                continue
            if cid in left_cids and cid in right_cids:
                increment = union_count - 1
                side = BlockSide.CLIQUE if block_pair.is_clique else BlockSide.BOTH
            elif cid in left_cids:
                increment = right_count
                side = BlockSide.LEFT
            else:
                increment = left_count
                side = BlockSide.RIGHT

            if increment <= 0:
                continue
            self.degrees[cid] += increment
            self.cluster_to_blocks[cid].append(ClusterBlockMembership(block_idx, side))

    def _pop_row(self, cluster_id: int) -> int:
        offset = int(self.removed[cluster_id])
        row_idx = int(self.cluster_map.members[cluster_id][offset])
        self.removed[cluster_id] += 1
        self.remaining[cluster_id] -= 1
        if self.remaining[cluster_id] == 0:
            self.degrees[cluster_id] = 0
        return row_idx

    def _update_neighbor_clusters(self, cluster_id: int) -> None:
        for membership in self.cluster_to_blocks[cluster_id]:
            block_pair = self.block_pairs[membership.block_pair_idx]
            affected = _affected_clusters(block_pair, membership.side)
            active_affected = affected[self.remaining[affected] > 0]
            if len(active_affected) == 0:
                continue
            self.degrees[active_affected] -= 1
            self.degrees[active_affected[self.degrees[active_affected] < 0]] = 0


def find_cluster_count_cover(n: int, violation_set: ViolationSetLike) -> list[int]:
    graph = ClusterCountGraph(n, violation_set)
    cover = []
    while graph.has_edges():
        cover.append(graph.pop_max_degree_vertex())
    return cover


def _combined_clusters(left_clusters: np.ndarray, right_clusters: np.ndarray) -> np.ndarray:
    if len(left_clusters) == 0:
        return right_clusters
    if len(right_clusters) == 0:
        return left_clusters
    if _cluster_sets_overlap(left_clusters, right_clusters):
        return np.unique(np.concatenate((left_clusters, right_clusters)))
    return np.concatenate((left_clusters, right_clusters))


def _cluster_sets_overlap(left_clusters: np.ndarray, right_clusters: np.ndarray) -> bool:
    if len(left_clusters) > len(right_clusters):
        left_clusters, right_clusters = right_clusters, left_clusters
    right = {int(cid) for cid in right_clusters}
    return any(int(cid) in right for cid in left_clusters)


def _cluster_count(cluster_sizes: np.ndarray, cluster_ids: np.ndarray) -> int:
    if len(cluster_ids) == 0:
        return 0
    return int(cluster_sizes[cluster_ids].sum())


def _affected_clusters(block_pair: ClusterBlockPair, side: BlockSide) -> np.ndarray:
    match side:
        case BlockSide.CLIQUE | BlockSide.BOTH:
            return block_pair.union_clusters
        case BlockSide.RIGHT:
            return block_pair.left_clusters
        case BlockSide.LEFT:
            return block_pair.right_clusters
