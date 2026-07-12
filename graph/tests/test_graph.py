from dataclasses import dataclass, field

import numpy as np

from graph import ConflictGraphBuilder
from graph.cluster_count_graph import find_cluster_count_cover


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


def violation_set(clusters: list[list[int]], violations: list[Violation]) -> tuple[int, ViolationSet]:
    cluster_indices = [np.asarray(rows, dtype=np.int64) for rows in clusters]
    n = max((int(row) for rows in cluster_indices for row in rows), default=-1) + 1
    row_to_cluster = np.full(n, -1, dtype=np.int64)
    for cluster_id, rows in enumerate(cluster_indices):
        row_to_cluster[rows] = cluster_id
    return n, ViolationSet(cluster_indices, row_to_cluster, violations)


def test_biclique_degrees_update_after_vertex_removal() -> None:
    n, violations = violation_set(
        clusters=[[0, 1], [2, 3, 4]],
        violations=[Violation(np.array([0]), np.array([1]))],
    )

    graph = ConflictGraphBuilder.build(n, violations)

    np.testing.assert_array_equal(graph.degrees, np.array([3, 3, 2, 2, 2]))
    assert graph.has_edges()

    graph.remove_vertex(0)

    np.testing.assert_array_equal(graph.degrees, np.array([0, 3, 1, 1, 1]))
    np.testing.assert_array_equal(graph.active, np.array([False, True, True, True, True]))
    assert graph.deleted[0]

    graph.remove_vertex(1)

    np.testing.assert_array_equal(graph.degrees, np.zeros(5, dtype=np.int64))
    assert not graph.has_edges()


def test_symmetric_violation_behaves_like_a_clique() -> None:
    n, violations = violation_set(
        clusters=[[0, 1, 2]],
        violations=[Violation(np.array([0]), np.array([0]), symmetric=True)],
    )

    graph = ConflictGraphBuilder.build(n, violations)

    np.testing.assert_array_equal(graph.degrees, np.array([2, 2, 2]))

    graph.remove_vertex(0)

    np.testing.assert_array_equal(graph.degrees, np.array([0, 1, 1]))
    assert graph.has_edges()

    graph.remove_vertex(1)

    np.testing.assert_array_equal(graph.degrees, np.array([0, 0, 0]))
    assert not graph.has_edges()


def test_overlapping_block_pair_counts_union_neighbors_once() -> None:
    n, violations = violation_set(
        clusters=[[0], [1], [2]],
        violations=[Violation(np.array([0, 1]), np.array([1, 2]))],
    )

    graph = ConflictGraphBuilder.build(n, violations)

    np.testing.assert_array_equal(graph.degrees, np.array([2, 2, 2]))

    graph.remove_vertex(1)

    np.testing.assert_array_equal(graph.degrees, np.array([1, 0, 1]))
    np.testing.assert_array_equal(graph.active, np.array([True, False, True]))
    assert graph.has_edges()


def test_random_edge_sampler_returns_live_edges() -> None:
    n, violations = violation_set(
        clusters=[[0, 1], [2, 3, 4]],
        violations=[Violation(np.array([0]), np.array([1]))],
    )
    graph = ConflictGraphBuilder.build(n, violations)
    rng = np.random.default_rng(7)
    expected_edges = {
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
    }

    for _ in range(25):
        edge = tuple(sorted(graph.pick_random_edge(rng)))
        assert edge in expected_edges


def test_cluster_count_cover_matches_biclique_greedy_result() -> None:
    n, violations = violation_set(
        clusters=[[0, 1], [2, 3, 4]],
        violations=[Violation(np.array([0]), np.array([1]))],
    )

    assert find_cluster_count_cover(n, violations) == [0, 1]


def test_cluster_count_cover_matches_clique_greedy_result() -> None:
    n, violations = violation_set(
        clusters=[[0, 1, 2]],
        violations=[Violation(np.array([0]), np.array([0]), symmetric=True)],
    )

    assert find_cluster_count_cover(n, violations) == [0, 1]


def test_cluster_count_cover_handles_overlapping_block_pair() -> None:
    n, violations = violation_set(
        clusters=[[0], [1], [2]],
        violations=[Violation(np.array([0, 1]), np.array([1, 2]))],
    )

    assert find_cluster_count_cover(n, violations) == [0, 1]
