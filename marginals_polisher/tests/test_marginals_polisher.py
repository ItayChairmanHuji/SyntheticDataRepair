from dataclasses import dataclass

import pandas as pd
import pytest

from marginals_error_calculator import MarginalsErrorCalculator
from marginals_polisher.marginals_polisher import get_max_deletions, polish


@dataclass
class FakeMarginal:
    attr: str
    value: str
    target: float

    def mask(self, df: pd.DataFrame):
        return df[self.attr].astype(str) == self.value


class FakeDataset:
    def __init__(self, data: pd.DataFrame, marginals: list[FakeMarginal] | None = None):
        self.data = data
        self.marginals = marginals

    def __len__(self) -> int:
        return len(self.data)


def test_polish_uses_row_positions_not_dataframe_index() -> None:
    data = pd.DataFrame({"x": ["a", "a", "b", "b"]}, index=[10, 20, 30, 40])
    dataset = FakeDataset(data, [FakeMarginal("x", "a", 0.25)])
    calc = MarginalsErrorCalculator(dataset)

    polished = polish(data, calc, max_deletions=1, batch_size=1)

    assert len(polished) == 3
    assert list(polished.index) == [0, 1, 2]
    assert (polished["x"] == "a").sum() == 1


def test_polish_does_not_commit_batch_that_worsens_error() -> None:
    data = pd.DataFrame({"x": ["a", "a", "b", "b"]})
    dataset = FakeDataset(data, [FakeMarginal("x", "a", 0.25)])
    calc = MarginalsErrorCalculator(dataset)

    polished = polish(data, calc, max_deletions=2, batch_size=2)

    assert len(polished) == 3
    assert (polished["x"] == "a").sum() == 1


def test_polish_without_marginals_returns_data_unchanged() -> None:
    data = pd.DataFrame({"x": ["a", "b", "c"]})
    calc = MarginalsErrorCalculator(FakeDataset(data))

    polished = polish(data, calc, max_deletions=2, batch_size=1)

    pd.testing.assert_frame_equal(polished, data)


def test_get_max_deletions_caps_at_all_but_one_row() -> None:
    dataset = FakeDataset(pd.DataFrame({"x": ["a", "b", "c"]}))

    assert get_max_deletions(dataset, 1.0) == 2


@pytest.mark.parametrize("deletion_budget", [-0.1, 1.1])
def test_get_max_deletions_rejects_invalid_budget(deletion_budget: float) -> None:
    dataset = FakeDataset(pd.DataFrame({"x": ["a", "b", "c"]}))

    with pytest.raises(ValueError):
        get_max_deletions(dataset, deletion_budget)
