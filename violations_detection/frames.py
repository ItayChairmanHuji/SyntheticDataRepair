from typing import Any, Iterable

import numpy as np
import pandas as pd

from denial_constraints import Predicate

from .predicate_eval import evaluate_frame_predicate


NULL_GROUP_KEY = object()


def filtered_frame(df: pd.DataFrame, predicates: tuple[Predicate, ...]) -> pd.DataFrame:
    mask = filter_mask(df, predicates)
    return df.loc[mask].assign(_cid=np.flatnonzero(mask))


def filter_mask(df: pd.DataFrame, predicates: tuple[Predicate, ...]) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for predicate in predicates:
        mask &= evaluate_frame_predicate(df, predicate)
    return mask


def matching_groups(left: pd.DataFrame, right: pd.DataFrame, keys: Iterable[str]):
    right_groups = groups_by(right, keys)
    for key, left_group in groups_by(left, keys).items():
        right_group = right_groups.get(key)
        if right_group is not None:
            yield left_group, right_group


def groups_by(frame: pd.DataFrame, keys: Iterable[str]) -> dict[tuple[Any, ...], pd.DataFrame]:
    keys = list(keys)
    if not keys:
        return {(): frame}

    by = keys[0] if len(keys) == 1 else keys
    return {
        normalize_group_key(key): group
        for key, group in frame.groupby(by, dropna=False, sort=False)
    }


def normalize_group_key(key) -> tuple[Any, ...]:
    values = key if isinstance(key, tuple) else (key,)
    return tuple(NULL_GROUP_KEY if pd.isna(value) else value for value in values)


def same_clusters(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return np.array_equal(
        left["_cid"].to_numpy(dtype=int),
        right["_cid"].to_numpy(dtype=int),
    )


def numeric_order_frame(frame: pd.DataFrame, attr: str, sort: bool = False) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "_cid": frame["_cid"].to_numpy(dtype=int),
            "_value": pd.to_numeric(frame[attr], errors="coerce").to_numpy(),
        }
    ).dropna(subset=["_value"])
    return out.sort_values("_value") if sort else out


def two_order_left_frame(frame: pd.DataFrame, first_attr: str, second_attr: str) -> pd.DataFrame:
    return two_order_frame(frame, first_attr, second_attr).dropna(subset=["_first", "_second"])


def two_order_right_frame(frame: pd.DataFrame, first_attr: str, second_attr: str) -> pd.DataFrame:
    return two_order_left_frame(frame, first_attr, second_attr).sort_values("_first")


def two_order_frame(frame: pd.DataFrame, first_attr: str, second_attr: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_cid": frame["_cid"].to_numpy(dtype=int),
            "_first": pd.to_numeric(frame[first_attr], errors="coerce").to_numpy(),
            "_second": pd.to_numeric(frame[second_attr], errors="coerce").to_numpy(),
        }
    )
