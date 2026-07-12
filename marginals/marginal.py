from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass
class Marginal:
    attrs: list[str]
    vals: list[Any]
    target: float

    def mask(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for attr, val in zip(self.attrs, self.vals):
            mask &= self._matches(df[attr], val)
        return mask

    def distance(self, df: pd.DataFrame, df2: pd.DataFrame | None = None) -> float:
        freq = self.mask(df).sum() / len(df)
        target = self.target if df2 is None else self.mask(df2).sum() / len(df2)
        return float(np.abs(target - freq))

    def error(self, df: pd.DataFrame, df2: pd.DataFrame | None = None) -> float:
        target = self.target if df2 is None else self.mask(df2).sum() / len(df2)
        return float(self.distance(df, df2) / (target + 1e-8))

    @staticmethod
    def _matches(series: pd.Series, val: Any) -> pd.Series:
        if pd.isna(val):
            return series.isna()

        numeric_val = Marginal._as_number(val)
        if numeric_val is not None:
            numeric_series = pd.to_numeric(series, errors="coerce")
            non_null = series.notna()
            all_non_null_values_are_numeric = bool(numeric_series[non_null].notna().all())

            if is_numeric_dtype(series) or all_non_null_values_are_numeric:
                return numeric_series == numeric_val

        return series.astype(str) == str(val)

    @staticmethod
    def _as_number(val: Any) -> float | None:
        numeric = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return None
        return float(numeric)
