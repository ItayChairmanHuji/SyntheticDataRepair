from itertools import combinations

import numpy as np

from denial_constraints import DenialConstraint

from .conflicts import add_block, add_internal_conflicts, add_order_range
from .constraint_plan import ConstraintPlan, OrderCondition
from .frames import (
    filtered_frame,
    groups_by,
    matching_groups,
    numeric_order_frame,
    same_clusters,
    two_order_left_frame,
    two_order_right_frame,
)
from .models import CompactData, Violation, ViolationSet
from .order_ranges import order_range
from .predicate_eval import compare_order


def add_not_equal_conflicts(
    df,
    plan: ConstraintPlan,
    violations: ViolationSet,
) -> None:
    attr = plan.not_equal[0]
    left = filtered_frame(df, plan.t1_filters)
    right = filtered_frame(df, plan.t2_filters)

    if same_clusters(left, right):
        for group in groups_by(left, plan.equalities).values():
            value_groups = groups_by(group, [attr]).values()
            for left_part, right_part in combinations(value_groups, 2):
                add_block(violations, left_part["_cid"], right_part["_cid"])
        return

    for left_group, right_group in matching_groups(left, right, plan.equalities):
        left_values = groups_by(left_group, [attr])
        right_values = groups_by(right_group, [attr])
        for left_key, left_part in left_values.items():
            for right_key, right_part in right_values.items():
                if left_key != right_key:
                    add_block(violations, left_part["_cid"], right_part["_cid"])


def add_order_conflicts(
    compact: CompactData,
    dc: DenialConstraint,
    plan: ConstraintPlan,
    violations: ViolationSet,
) -> None:
    left = filtered_frame(compact.df, plan.t1_filters)
    right = filtered_frame(compact.df, plan.t2_filters)

    for left_group, right_group in matching_groups(left, right, plan.equalities):
        if len(plan.orders) == 1:
            add_single_order_group(left_group, right_group, plan.orders[0], violations)
        else:
            add_two_order_group(left_group, right_group, plan.orders, violations)

    add_internal_conflicts(compact, dc, violations)


def add_single_order_group(
    left,
    right,
    order: OrderCondition,
    violations: ViolationSet,
) -> None:
    left_values = numeric_order_frame(left, order.attr)
    right_values = numeric_order_frame(right, order.attr, sort=True)
    right_ids = right_values["_cid"].to_numpy(dtype=int)
    values = right_values["_value"].to_numpy()

    for cid, value in left_values[["_cid", "_value"]].itertuples(index=False):
        start, end = order_range(order.operator, value, values)
        add_order_range(violations, int(cid), start, end, right_ids)


def add_two_order_group(
    left,
    right,
    orders: tuple[OrderCondition, ...],
    violations: ViolationSet,
) -> None:
    first, second = orders
    left_values = two_order_left_frame(left, first.attr, second.attr)
    right_values = two_order_right_frame(right, first.attr, second.attr)
    right_ids = right_values["_cid"].to_numpy(dtype=int)
    first_values = right_values["_first"].to_numpy()
    second_values = right_values["_second"].to_numpy()

    for cid, first_value, second_value in left_values.itertuples(index=False):
        start, end = order_range(first.operator, first_value, first_values)
        if start >= end:
            continue

        candidate_ids = right_ids[start:end]
        mask = candidate_ids != int(cid)
        mask &= compare_order(second_value, second.operator, second_values[start:end])
        if mask.any():
            violations.violations.append(Violation(np.array([int(cid)]), candidate_ids[mask]))


def add_all_pairs(
    compact: CompactData,
    dc: DenialConstraint,
    plan: ConstraintPlan,
    violations: ViolationSet,
) -> None:
    left = filtered_frame(compact.df, plan.t1_filters)
    right = filtered_frame(compact.df, plan.t2_filters)

    if same_clusters(left, right):
        for group in groups_by(left, plan.equalities).values():
            ids = group["_cid"].to_numpy(dtype=int)
            if len(ids):
                violations.violations.append(Violation(ids, ids, symmetric=True))
        return

    for left_group, right_group in matching_groups(left, right, plan.equalities):
        add_block(violations, left_group["_cid"], right_group["_cid"])
    add_internal_conflicts(compact, dc, violations)
