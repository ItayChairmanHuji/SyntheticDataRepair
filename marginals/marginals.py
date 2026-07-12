from pathlib import Path

import numpy as np
import pandas as pd

from marginals.marginal import Marginal


class Marginals:
    def __init__(self, path: Path):
        marginals = pd.read_csv(path)
        self.marginals = []
        for _, row in marginals.iterrows():
            attrs = [row["attr1"], row["attr2"]]
            vals = [row["value1"], row["value2"]]
            target = row["probability"]
            self.marginals.append(Marginal(attrs, vals, target))

    def filter(self, size: int):
        self.marginals = self.marginals[:size]

    def __iter__(self):
        return iter(self.marginals)

    def __len__(self):
        return len(self.marginals)

    def error(self, df: pd.DataFrame, df2: pd.DataFrame | None = None):
        return np.array([m.error(df, df2) for m in self.marginals]).mean()

    def distance(self, df: pd.DataFrame, df2: pd.DataFrame | None = None):
        return np.array([m.distance(df, df2) for m in self.marginals]).mean()

    def cumulated_error(self, df: pd.DataFrame, df2: pd.DataFrame | None = None):
        errors = np.array([m.error(df, df2) for m in self.marginals])
        return np.array([errors[:i].mean() for i in range(10, len(errors) + 1, 10)])

    def cumulated_distance(self, df: pd.DataFrame, df2: pd.DataFrame | None = None):
        distances = np.array([m.distance(df, df2) for m in self.marginals])
        return np.array([distances[:i].mean() for i in range(10, len(distances) + 1, 10)])

    @staticmethod
    def max_marginals(dataset_name: str) -> int:
        marginals = pd.read_csv(Marginals.BASE_PATH / f"{dataset_name}.csv")
        return len(marginals)
