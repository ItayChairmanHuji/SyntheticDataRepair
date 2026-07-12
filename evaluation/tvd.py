from itertools import combinations

import pandas as pd

from dataset.dataset import Dataset


def average_two_way_tvd(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    num_pairs = 0
    total_tvd = 0.0
    columns = df1.columns
    for col_a, col_b in combinations(columns, 2):
        dist1 = df1.groupby([col_a, col_b], dropna=False).size() / len(df1)
        dist2 = df2.groupby([col_a, col_b], dropna=False).size() / len(df2)

        all_keys = dist1.index.union(dist2.index)

        dist1 = dist1.reindex(all_keys, fill_value=0)
        dist2 = dist2.reindex(all_keys, fill_value=0)

        tvd = 0.5 * (dist1 - dist2).abs().sum()

        total_tvd += tvd
        num_pairs += 1
    return float(total_tvd / num_pairs) if num_pairs > 0 else 0.0


def tvd(dataset: Dataset):
    private = dataset.load_private_data()
    return average_two_way_tvd(dataset.data, private.data)
