import numpy as np
import pandas as pd

from denial_constraints import Predicate, Side

from .predicates import EQUALITY_OPERATORS, INEQUALITY_OPERATORS, is_number, literal


def evaluate_frame_predicate(df: pd.DataFrame, predicate: Predicate) -> np.ndarray:
    result = compare_operands(
        frame_operand(df, predicate.left),
        predicate.opr,
        frame_operand(df, predicate.right),
    )
    if isinstance(result, pd.Series):
        return result.fillna(False).to_numpy(dtype=bool)
    return np.full(len(df), bool(result), dtype=bool)


def evaluate_row_predicate(row: pd.Series, predicate: Predicate) -> bool:
    return bool(
        compare_scalars(
            row_operand(row, predicate.left),
            predicate.opr,
            row_operand(row, predicate.right),
        )
    )


def compare_order(left, operator: str, right):
    match operator:
        case ">":
            return left > right
        case ">=":
            return left >= right
        case "<":
            return left < right
        case "<=":
            return left <= right
    raise ValueError(f"Unsupported order operator: {operator}")


def compare_operands(left, operator: str, right):
    if operator in EQUALITY_OPERATORS | INEQUALITY_OPERATORS:
        left, right = align_numeric_literal(left, right)
        equal = null_equal(left, right)
        return equal if operator in EQUALITY_OPERATORS else invert(equal)

    return compare_order(numeric(left), operator, numeric(right))


def compare_scalars(left, operator: str, right) -> bool:
    left, right = align_numeric_scalars(left, right)
    if operator in EQUALITY_OPERATORS:
        return same_value(left, right)
    if operator in INEQUALITY_OPERATORS:
        return not same_value(left, right)

    left = numeric(left)
    right = numeric(right)
    if pd.isna(left) or pd.isna(right):
        return False
    return bool(compare_order(left, operator, right))


def frame_operand(df: pd.DataFrame, side: Side):
    return literal(side.attr) if side.is_value else df[side.attr]


def row_operand(row: pd.Series, side: Side):
    return literal(side.attr) if side.is_value else row[side.attr]


def align_numeric_literal(left, right):
    if is_number(left) and isinstance(right, pd.Series):
        return left, pd.to_numeric(right, errors="coerce")
    if is_number(right) and isinstance(left, pd.Series):
        return pd.to_numeric(left, errors="coerce"), right
    return left, right


def align_numeric_scalars(left, right):
    if is_number(left) or is_number(right):
        return numeric(left), numeric(right)
    return left, right


def null_equal(left, right):
    equal = equals(left, right)
    return equal | (pd.isna(left) & pd.isna(right))


def equals(left, right):
    if isinstance(left, pd.Series):
        return left.eq(right)
    if isinstance(right, pd.Series):
        return right.eq(left)
    return left == right


def numeric(value):
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce")
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def same_value(left, right) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    return bool(left == right)


def invert(value):
    return ~value if isinstance(value, pd.Series) else not value
