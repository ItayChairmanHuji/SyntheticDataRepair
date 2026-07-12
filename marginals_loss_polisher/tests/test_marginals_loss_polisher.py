from dataclasses import dataclass

import pandas as pd
import pytest

from marginals_error_calculator import MarginalsErrorCalculator
from marginals_loss_polisher.marginals_loss_polisher import get_loss, get_max_deletions, polish


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


def test_get_loss_combines_size_and_marginal_terms() -> None:
    loss = get_loss(original_size=10, current_size=9, marginals_error=0.4, alpha=0.25)

    assert loss == pytest.approx(0.325)


def test_get_loss_uses_only_size_term_when_no_marginals_are_present() -> None:
    loss = get_loss(original_size=10, current_size=8, marginals_error=0.4, alpha=0.75, has_marginals=False)

    assert loss == pytest.approx(0.15)


def test_polish_deletes_when_marginal_gain_outweighs_size_penalty() -> None:
    data = pd.DataFrame({"x": ["a", "a", "b", "b"]}, index=[10, 20, 30, 40])
    dataset = FakeDataset(data, [FakeMarginal("x", "a", 0.25)])
    calc = MarginalsErrorCalculator(dataset)

    polished = polish(data, calc, max_deletions=1, batch_size=1, alpha=0.5)

    assert len(polished) == 3
    assert list(polished.index) == [0, 1, 2]
    assert (polished["x"] == "a").sum() == 1


def test_polish_uses_row_positions_not_dataframe_index() -> None:
    data = pd.DataFrame({"x": ["a", "a", "b", "b"]}, index=[10, 20, 30, 40])
    dataset = FakeDataset(data, [FakeMarginal("x", "a", 0.25)])
    calc = MarginalsErrorCalculator(dataset)

    polished = polish(data, calc, max_deletions=1, batch_size=1, alpha=0.0)

    assert len(polished) == 3
    assert list(polished.index) == [0, 1, 2]
    assert (polished["x"] == "a").sum() == 1


def test_polish_without_marginals_and_zero_alpha_returns_data_unchanged() -> None:
    data = pd.DataFrame({"x": ["a", "b", "c"]})
    calc = MarginalsErrorCalculator(FakeDataset(data))

    polished = polish(data, calc, max_deletions=2, batch_size=1, alpha=0.0)

    pd.testing.assert_frame_equal(polished, data)


def test_get_max_deletions_caps_at_all_but_one_row() -> None:
    dataset = FakeDataset(pd.DataFrame({"x": ["a", "b", "c"]}))

    assert get_max_deletions(dataset, 1.0) == 2


@pytest.mark.parametrize("deletion_budget", [-0.1, 1.1])
def test_get_max_deletions_rejects_invalid_budget(deletion_budget: float) -> None:
    dataset = FakeDataset(pd.DataFrame({"x": ["a", "b", "c"]}))

    with pytest.raises(ValueError):
        get_max_deletions(dataset, deletion_budget)
