import numpy as np

from denial_constraints import DenialConstraint

from .models import CompactData, Violation, ViolationSet
from .predicate_eval import evaluate_row_predicate


def add_block(violations: ViolationSet, left, right) -> None:
    left_ids = np.asarray(left, dtype=int)
    right_ids = np.asarray(right, dtype=int)
    if len(left_ids) and len(right_ids):
        violations.violations.append(Violation(left_ids, right_ids))


def add_order_range(
    violations: ViolationSet,
    cid: int,
    start: int,
    end: int,
    right_ids: np.ndarray,
) -> None:
    if start >= end:
        return

    matches = np.flatnonzero(right_ids[start:end] == cid)
    if len(matches) == 0:
        violations.violations.append(Violation(np.array([cid]), right_ids[start:end]))
        return

    split = start + int(matches[0])
    if start < split:
        violations.violations.append(Violation(np.array([cid]), right_ids[start:split]))
    if split + 1 < end:
        violations.violations.append(Violation(np.array([cid]), right_ids[split + 1 : end]))


def add_internal_conflicts(
    compact: CompactData,
    dc: DenialConstraint,
    violations: ViolationSet,
) -> None:
    for cid, row in compact.df.iterrows():
        if len(compact._compact_to_dense[int(cid)]) <= 1:
            continue
        if all(evaluate_row_predicate(row, predicate) for predicate in dc.predicates):
            cluster = np.array([int(cid)])
            violations.violations.append(Violation(cluster, cluster, symmetric=True))
