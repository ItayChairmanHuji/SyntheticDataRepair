from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("gurobipy")

from denial_constraints import DenialConstraint, DenialConstraints
from loss_ilp_repair.loss_ilp_repair import repair
from marginals.marginal import Marginal


@dataclass
class MarginalSet:
    marginals: list[Marginal]

    def __iter__(self):
        return iter(self.marginals)


def constraints(*raw_constraints: str) -> DenialConstraints:
    return DenialConstraints([DenialConstraint.from_string(raw) for raw in raw_constraints])


def marginal(attrs: list[str], vals: list[Any], target: float) -> Marginal:
    return Marginal(attrs, vals, target)


def test_repair_solves_dc_conflict_and_prefers_lower_marginal_error() -> None:
    data = pd.DataFrame({"A": [1, 1, 2], "B": ["x", "y", "x"]})
    dcs = constraints("not(t1.A=t2.A&t1.B!=t2.B)")
    marginals = MarginalSet([marginal(["A", "B"], [1, "x"], 0.5)])

    repaired = repair(data, dcs, marginals, alpha=0.5)

    assert repaired.index.tolist() == [0, 2]


def test_repair_compacts_with_marginal_attributes_before_optimizing() -> None:
    data = pd.DataFrame(
        {
            "A": [1, 1, 2],
            "C": ["target", "other", "other"],
        }
    )
    dcs = constraints("not(t1.A=t2.A)")
    marginals = MarginalSet([marginal(["C"], ["target"], 0.5)])

    repaired = repair(data, dcs, marginals, alpha=0.5)

    assert repaired.index.tolist() == [0, 2]
