from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .block_pair import BlockMembership, BlockPair


@dataclass(slots=True)
class Graph:
    n: int
    degrees: np.ndarray
    active: np.ndarray
    deleted: np.ndarray
    block_pairs: list[BlockPair]
    cluster_to_blocks: list[list[BlockMembership]]
    row_to_cluster: np.ndarray
    _active_vertices: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._active_vertices = int(np.count_nonzero(self.active))

    def has_edges(self) -> bool:
        return self._active_vertices > 0

    def degree(self, indices: int | np.ndarray) -> int | np.ndarray:
        values = self.degrees[indices]
        if np.isscalar(values):
            return int(values)
        return values

    def remove_vertex(self, row_idx: int) -> None:
        if self.deleted[row_idx]:
            return

        if not self.active[row_idx]:
            self.deleted[row_idx] = True
            self.degrees[row_idx] = 0
            return

        self._update_neighbors(row_idx)
        self.deleted[row_idx] = True
        self.active[row_idx] = False
        self.degrees[row_idx] = 0
        self._active_vertices -= 1

    def pick_random_edge(self, rng: np.random.Generator | None = None) -> tuple[int, int]:
        if not self.has_edges():
            raise ValueError("No edges left in graph")

        random = rng or np.random
        active_indices = np.flatnonzero(self.active)
        active_degrees = self.degrees[active_indices]
        total_degree = int(active_degrees.sum())
        if total_degree <= 0:
            raise ValueError("No edges left in graph")

        u = int(random.choice(active_indices, p=active_degrees / total_degree))
        neighbor_offset = _random_integer(random, int(self.degrees[u]))

        for membership in self._memberships_for_row(u):
            active_neighbors = self._active_neighbors(u, membership)
            neighbor_count = len(active_neighbors)
            if neighbor_offset < neighbor_count:
                return u, int(random.choice(active_neighbors))
            neighbor_offset -= neighbor_count

        raise ValueError(f"Vertex {u} has degree {self.degrees[u]} but no active neighbors.")

    def _update_neighbors(self, row_idx: int) -> None:
        for membership in self._memberships_for_row(row_idx):
            active_neighbors = self._active_neighbors(row_idx, membership)
            if len(active_neighbors) == 0:
                continue

            self.degrees[active_neighbors] -= 1
            became_inactive = active_neighbors[self.degrees[active_neighbors] <= 0]
            if len(became_inactive) > 0:
                self.degrees[became_inactive] = 0
                self.active[became_inactive] = False
                self._active_vertices -= int(len(became_inactive))

    def _memberships_for_row(self, row_idx: int) -> list[BlockMembership]:
        cluster_id = int(self.row_to_cluster[row_idx])
        if cluster_id < 0 or cluster_id >= len(self.cluster_to_blocks):
            return []
        return self.cluster_to_blocks[cluster_id]

    def _active_neighbors(self, row_idx: int, membership: BlockMembership) -> np.ndarray:
        block_pair = self.block_pairs[membership.block_pair_idx]
        neighbors = membership.neighbor_members(block_pair)
        return neighbors[self.active[neighbors] & (neighbors != row_idx)]


def _random_integer(rng, high: int) -> int:
    if hasattr(rng, "integers"):
        return int(rng.integers(high))
    return int(rng.randint(high))
