from typing import Any, Tuple

import pandas as pd

from dataset.dataset import Dataset
from marginals.marginals import Marginals

Measurement = Tuple[str, str, Any, Any]


def average_measurement_distance(df1: pd.DataFrame, df2: pd.DataFrame, marginals: Marginals) -> float:
    total_distance = 0.0
    for marginal in marginals:
        attr1, attr2, val1, val2 = marginal.attrs[0], marginal.attrs[1], marginal.vals[0], marginal.vals[1]
        p1 = ((df1[attr1] == val1) & (df1[attr2] == val2)).mean()
        p2 = ((df2[attr1] == val1) & (df2[attr2] == val2)).mean()

        total_distance += abs(p1 - p2)

    return float(total_distance / len(marginals))


def marginals_tvd(dataset: Dataset) -> float:
    private = dataset.load_private_data()
    if dataset.marginals is None:
        raise ValueError("Dataset does not have marginals.")
    return average_measurement_distance(dataset.data, private.data, dataset.marginals)
