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
