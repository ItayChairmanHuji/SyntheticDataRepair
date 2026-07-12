from numbers import Number

from denial_constraints import Predicate


EQUALITY_OPERATORS = {"=", "=="}
INEQUALITY_OPERATORS = {"!=", "<>"}
ORDER_OPERATORS = {">", ">=", "<", "<="}


def tuple_indexes(predicate: Predicate) -> tuple[int, ...]:
    indexes = {
        side.index
        for side in (predicate.left, predicate.right)
        if not side.is_value and side.index is not None
    }
    return tuple(sorted(indexes))


def is_same_attr_cross(predicate: Predicate, operators: set[str]) -> bool:
    indexes = tuple_indexes(predicate)
    return (
        len(indexes) == 2
        and predicate.opr in operators
        and not predicate.left.is_value
        and not predicate.right.is_value
        and predicate.left.attr == predicate.right.attr
    )


def reverse_operator(operator: str) -> str:
    return {">": "<", ">=": "<=", "<": ">", "<=": ">="}[operator]


def literal(value):
    if not isinstance(value, str):
        return value
    try:
        return float(value)
    except ValueError:
        return value


def is_number(value) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)
