import pandas as pd

from denial_constraints import DenialConstraint, DenialConstraints
from violations_detection.models import compact_data
from violations_detection import ViolationFinder, find_violations


class InMemoryDataset:
    def __init__(self, data: pd.DataFrame, dcs: DenialConstraints):
        self.data = data
        self.dcs = dcs
        self._compact = {}

    def compact(self, attributes: list[str] | None = None):
        attrs = sorted(attributes) if attributes else sorted(self.dcs.attrs)
        key = tuple(attrs)
        if key not in self._compact:
            self._compact[key] = compact_data(self.data, attrs)
        return self._compact[key]


def constraints(*raw_constraints: str) -> DenialConstraints:
    return DenialConstraints([DenialConstraint.from_string(raw) for raw in raw_constraints])


def dataset(data: pd.DataFrame, dcs: DenialConstraints) -> InMemoryDataset:
    return InMemoryDataset(data, dcs)


def violations_for(data: pd.DataFrame, raw_constraint: str):
    return find_violations(dataset(data, constraints(raw_constraint)))


def edges(violations) -> set[tuple[int, int]]:
    frame = violations.to_dataframe()
    if frame.empty:
        return set()
    return {
        tuple(sorted((int(row.idx1), int(row.idx2))))
        for row in frame.itertuples()
    }


def test_public_function_finds_fd_violations() -> None:
    data = pd.DataFrame({"A": [1, 1, 2, 2], "B": ["x", "y", "x", "x"]})

    violations = violations_for(data, "not(t1.A=t2.A&t1.B!=t2.B)")

    assert edges(violations) == {(0, 1)}


def test_fd_groups_nullable_keys() -> None:
    data = pd.DataFrame({"A": [1, 1, None, None], "B": ["x", "y", "x", "y"]})

    violations = ViolationFinder().find_violations(
        dataset(data, constraints("not(t1.A=t2.A&t1.B!=t2.B)"))
    )

    assert edges(violations) == {(0, 1), (2, 3)}


def test_conditional_constant_filters_tuple_sides() -> None:
    data = pd.DataFrame(
        {
            "A": [1, 1, 1],
            "B": ["x", "y", "x"],
            "C": ["v1", "v2", "v1"],
        }
    )

    violations = violations_for(data, "not(t1.A=t2.A&t1.B!=t2.B&t1.C='v1')")

    assert edges(violations) == {(0, 1), (1, 2)}


def test_filter_only_constraints_connect_matching_tuples() -> None:
    data = pd.DataFrame({"AGEP": [0, 0, 1], "SCHL": [0, 1, 0]})

    violations = violations_for(data, "not(t1.AGEP=0&t2.AGEP=0&t1.SCHL=0&t2.SCHL!=0)")

    assert edges(violations) == {(0, 1)}


def test_single_order_uses_sorted_range_lookup() -> None:
    data = pd.DataFrame({"A": [10, 20, 30]})

    violations = violations_for(data, "not(t1.A>t2.A)")

    assert edges(violations) == {(0, 1), (0, 2), (1, 2)}


def test_two_order_filters_candidates_with_second_order() -> None:
    data = pd.DataFrame({"A": [10, 20, 15], "B": [100, 50, 150]})

    violations = violations_for(data, "not(t1.A>t2.A&t1.B<t2.B)")

    assert edges(violations) == {(0, 1), (1, 2)}


def test_grouped_two_order_stays_on_fast_path() -> None:
    data = pd.DataFrame(
        {
            "State": ["CA", "CA", "TX", "TX"],
            "Salary": [100, 80, 100, 90],
            "Rate": [0.1, 0.2, 0.1, 0.05],
        }
    )

    violations = violations_for(
        data,
        "not(t1.State=t2.State&t1.Salary>t2.Salary&t1.Rate<t2.Rate)",
    )

    assert edges(violations) == {(0, 1)}


def test_equality_only_constraint_reports_internal_cluster_edges() -> None:
    data = pd.DataFrame({"A": [1, 1, 1, 2]})

    violations = violations_for(data, "not(t1.A=t2.A)")

    assert edges(violations) == {(0, 1), (0, 2), (1, 2)}
    assert len(violations) == 3
