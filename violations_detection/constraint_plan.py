from dataclasses import dataclass

from denial_constraints import DenialConstraint, Predicate

from .predicates import (
    EQUALITY_OPERATORS,
    INEQUALITY_OPERATORS,
    ORDER_OPERATORS,
    is_same_attr_cross,
    reverse_operator,
    tuple_indexes,
)


@dataclass(frozen=True)
class OrderCondition:
    attr: str
    operator: str


@dataclass(frozen=True)
class ConstraintPlan:
    equalities: tuple[str, ...]
    not_equal: tuple[str, ...]
    orders: tuple[OrderCondition, ...]
    t1_filters: tuple[Predicate, ...]
    t2_filters: tuple[Predicate, ...]
    complex_predicates: tuple[Predicate, ...]

    @property
    def uses_not_equal_fast_path(self) -> bool:
        return len(self.not_equal) == 1 and not self.orders and not self.complex_predicates

    @property
    def uses_order_fast_path(self) -> bool:
        return not self.not_equal and not self.complex_predicates and 1 <= len(self.orders) <= 2

    @property
    def uses_all_pairs_fast_path(self) -> bool:
        return not self.not_equal and not self.orders and not self.complex_predicates


def plan_constraint(dc: DenialConstraint) -> ConstraintPlan:
    equalities: list[str] = []
    not_equal: list[str] = []
    orders: list[OrderCondition] = []
    t1_filters: list[Predicate] = []
    t2_filters: list[Predicate] = []
    complex_predicates: list[Predicate] = []

    for predicate in dc.predicates:
        indexes = tuple_indexes(predicate)
        if len(indexes) == 1:
            (index,) = indexes
            (t1_filters if index == 1 else t2_filters).append(predicate)
        elif is_same_attr_cross(predicate, EQUALITY_OPERATORS):
            equalities.append(predicate.left.attr)
        elif is_same_attr_cross(predicate, INEQUALITY_OPERATORS):
            not_equal.append(predicate.left.attr)
        elif is_same_attr_cross(predicate, ORDER_OPERATORS):
            orders.append(_order_condition(predicate))
        else:
            complex_predicates.append(predicate)

    return ConstraintPlan(
        equalities=tuple(equalities),
        not_equal=tuple(not_equal),
        orders=tuple(orders),
        t1_filters=tuple(t1_filters),
        t2_filters=tuple(t2_filters),
        complex_predicates=tuple(complex_predicates),
    )


def _order_condition(predicate: Predicate) -> OrderCondition:
    if predicate.left.index == 1 and predicate.right.index == 2:
        return OrderCondition(attr=predicate.left.attr, operator=predicate.opr)
    return OrderCondition(attr=predicate.left.attr, operator=reverse_operator(predicate.opr))
