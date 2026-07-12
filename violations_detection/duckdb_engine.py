import logging
from numbers import Number
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from denial_constraints import DenialConstraint, Predicate, Side

from .models import CompactData, Violation, ViolationSet, violation_set_for
from .predicates import tuple_indexes


logger = logging.getLogger(__name__)


class DuckDBEngine:
    def __init__(self):
        self.connection = duckdb.connect(database=":memory:")

    def find_violations_for_compact(self, dc: DenialConstraint, compact: CompactData) -> ViolationSet:
        violations = violation_set_for(compact)
        self._register_clusters(dc, compact)
        self._add_pairwise_conflicts(dc, violations)
        self._add_internal_conflicts(dc, compact, violations)
        return violations


    def _register_clusters(self, dc: DenialConstraint, compact: CompactData) -> None:
        frame = compact.df[list(dc.attrs)].copy()
        frame["_cid"] = np.arange(len(compact.df))
        self.connection.register("clusters", frame)

    def _add_pairwise_conflicts(self, dc: DenialConstraint, violations: ViolationSet) -> None:
        result = self._query(_pairwise_sql(dc), dc)
        for cid1, group in result.groupby("cid1"):
            violations.violations.append(
                Violation(np.array([int(cid1)]), group["cid2"].to_numpy(dtype=int))
            )

    def _add_internal_conflicts(
        self,
        dc: DenialConstraint,
        compact: CompactData,
        violations: ViolationSet,
    ) -> None:
        result = self._query(_internal_sql(dc), dc)
        for cid in result.get("_cid", []):
            cid = int(cid)
            if len(compact._compact_to_dense[cid]) > 1:
                cluster = np.array([cid])
                violations.violations.append(Violation(cluster, cluster, symmetric=True))

    def _query(self, sql: str, dc: DenialConstraint) -> pd.DataFrame:
        try:
            return self.connection.execute(sql).fetchdf()
        except Exception:
            logger.exception("DuckDB error for DC %s\nSQL:\n%s", dc.to_string(), sql)
            return pd.DataFrame()


def _pairwise_sql(dc: DenialConstraint) -> str:
    joins, predicates = _partition_predicates(dc.predicates)
    return f"""
        SELECT DISTINCT
            LEAST(t1._cid, t2._cid) AS cid1,
            GREATEST(t1._cid, t2._cid) AS cid2
        FROM {_from_clause(joins)}
        WHERE {_where_clause(predicates)}
          AND t1._cid != t2._cid
    """


def _internal_sql(dc: DenialConstraint) -> str:
    predicates = [_predicate_sql(predicate, "t1", "t1") for predicate in dc.predicates]
    return f"SELECT _cid FROM clusters t1 WHERE {_where_clause(predicates)}"


def _partition_predicates(predicates: list[Predicate]) -> tuple[list[Predicate], list[str]]:
    joins = []
    where = []
    for predicate in predicates:
        if _is_join_equality(predicate):
            joins.append(predicate)
        else:
            where.append(_where_sql(predicate))
    return joins, where


def _where_sql(predicate: Predicate) -> str:
    indexes = tuple_indexes(predicate)
    if len(indexes) <= 1:
        table = "t1" if not indexes or indexes[0] == 1 else "t2"
        return _predicate_sql(predicate, table, table)
    return _predicate_sql(predicate, "t1", "t2")


def _from_clause(joins: list[Predicate]) -> str:
    if not joins:
        return "clusters t1, clusters t2"

    conditions = [
        f"t1.{_identifier(predicate.left.attr)} IS NOT DISTINCT FROM "
        f"t2.{_identifier(predicate.right.attr)}"
        for predicate in joins
    ]
    return f"clusters t1 JOIN clusters t2 ON {' AND '.join(conditions)}"


def _where_clause(predicates: list[str]) -> str:
    return " AND ".join(predicates) if predicates else "TRUE"


def _is_join_equality(predicate: Predicate) -> bool:
    return (
        predicate.opr in {"=", "=="}
        and tuple_indexes(predicate) == (1, 2)
        and not predicate.left.is_value
        and not predicate.right.is_value
        and predicate.left.attr == predicate.right.attr
    )


def _predicate_sql(predicate: Predicate, t1_name: str, t2_name: str) -> str:
    left = _side_sql(predicate.left, t1_name, t2_name)
    right = _side_sql(predicate.right, t1_name, t2_name)
    match predicate.opr:
        case "=" | "==":
            return f"{left} IS NOT DISTINCT FROM {right}"
        case "!=" | "<>":
            return f"{left} IS DISTINCT FROM {right}"
        case operator:
            return f"{left} {operator} {right}"


def _side_sql(side: Side, t1_name: str, t2_name: str) -> str:
    if side.is_value:
        return _literal_sql(side.attr)
    table = t1_name if side.index == 1 else t2_name
    return f"{table}.{_identifier(side.attr)}"


def _identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _literal_sql(value) -> str:
    if value is None or pd.isna(value):
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, Number):
        return str(value)
    if isinstance(value, str):
        try:
            float(value)
            return value
        except ValueError:
            return "'" + value.replace("'", "''") + "'"
    return "'" + str(value).replace("'", "''") + "'"
