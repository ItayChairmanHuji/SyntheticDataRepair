from __future__ import annotations

import pandas as pd

from denial_constraints import DenialConstraint
from denial_constraints.denial_constraints import DenialConstraints

from .constraint_plan import plan_constraint
from .duckdb_engine import DuckDBEngine
from .fast_paths import add_all_pairs, add_not_equal_conflicts, add_order_conflicts
from .models import CompactData, ViolationSet, compact_data


class ViolationFinder:
    def __init__(self, duckdb_engine: DuckDBEngine | None = None):
        self._duckdb_engine = duckdb_engine

    def find_violations(self, data: pd.DataFrame, dcs: DenialConstraints | None = None) -> ViolationSet:
        if dcs is None:
            dcs = data.dcs
            compact = data.compact(list(dcs.attrs))
        else:
            compact = compact_data(data, list(dcs.attrs))
        return self.find_compact_violations(compact, dcs.constraints)

    def find_compact_violations(self, compact: CompactData, constraints: list[DenialConstraint]) -> ViolationSet:
        violations = compact.to_violation_set()
        for dc in constraints:
            self._add_constraint_violations(compact, dc, violations)
        return violations

    def _add_constraint_violations(self, compact: CompactData, dc: DenialConstraint, violations: ViolationSet) -> None:
        plan = plan_constraint(dc)
        if plan.uses_not_equal_fast_path:
            add_not_equal_conflicts(compact.df, plan, violations)
        elif plan.uses_order_fast_path:
            add_order_conflicts(compact, dc, plan, violations)
        elif plan.uses_all_pairs_fast_path:
            add_all_pairs(compact, dc, plan, violations)
        else:
            fallback = self._duckdb.find_violations_for_compact(dc, compact)
            violations.violations.extend(fallback.violations)

    @property
    def _duckdb(self) -> DuckDBEngine:
        if self._duckdb_engine is None:
            self._duckdb_engine = DuckDBEngine()
        return self._duckdb_engine


def find_violations(data: pd.DataFrame, dcs: DenialConstraints | None = None) -> ViolationSet:
    return ViolationFinder().find_violations(data, dcs)
