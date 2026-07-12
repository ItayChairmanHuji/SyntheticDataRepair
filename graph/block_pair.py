from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .cluster_map import ClusterMap
from .types import ViolationLike


class BlockSide(Enum):
    CLIQUE = "clique"
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class BlockMembership:
    block_pair_idx: int
    side: BlockSide

    def neighbor_members(self, block_pair: BlockPair) -> np.ndarray:
        match self.side:
            case BlockSide.CLIQUE | BlockSide.BOTH:
                return block_pair.union_members
            case BlockSide.RIGHT:
                return block_pair.left_members
            case BlockSide.LEFT:
                return block_pair.right_members

    def affected_members(self, block_pair: BlockPair) -> np.ndarray:
        return self.neighbor_members(block_pair)


@dataclass(frozen=True, slots=True)
class BlockPair:
    left_clusters: np.ndarray
    right_clusters: np.ndarray
    left_members: np.ndarray
    right_members: np.ndarray
    union_members: np.ndarray
    is_clique: bool = False

    @classmethod
    def from_conflict(cls, conflict: ViolationLike, clusters: ClusterMap) -> BlockPair:
        left_clusters = _as_cluster_ids(conflict.left)
        right_clusters = left_clusters if conflict.symmetric else _as_cluster_ids(conflict.right)

        left_members = clusters.members_for(left_clusters)
        right_members = left_members if conflict.symmetric else clusters.members_for(right_clusters)
        union_members = (
            left_members
            if conflict.symmetric
            else _combined_members(left_clusters, right_clusters, left_members, right_members)
        )

        return cls(
            left_clusters=left_clusters,
            right_clusters=right_clusters,
            left_members=left_members,
            right_members=right_members,
            union_members=union_members,
            is_clique=conflict.symmetric,
        )


def _as_cluster_ids(cluster_ids: np.ndarray) -> np.ndarray:
    return np.asarray(cluster_ids, dtype=np.int64)


def _combined_members(
    left_clusters: np.ndarray,
    right_clusters: np.ndarray,
    left_members: np.ndarray,
    right_members: np.ndarray,
) -> np.ndarray:
    if len(left_members) == 0:
        return right_members
    if len(right_members) == 0:
        return left_members
    if _cluster_sets_overlap(left_clusters, right_clusters):
        return np.unique(np.concatenate((left_members, right_members)))
    return np.concatenate((left_members, right_members))


def _cluster_sets_overlap(left_clusters: np.ndarray, right_clusters: np.ndarray) -> bool:
    if len(left_clusters) > len(right_clusters):
        left_clusters, right_clusters = right_clusters, left_clusters
    right = {int(cid) for cid in right_clusters}
    return any(int(cid) in right for cid in left_clusters)
