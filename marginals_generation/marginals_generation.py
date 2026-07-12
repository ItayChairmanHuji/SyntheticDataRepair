import itertools
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig


def zcdp_gaussian_sigma(num_rows: int, rho: float) -> float:
    sensitivity = 1 / num_rows
    return sensitivity / np.sqrt(2 * rho)


def compute_two_way_marginals(data: pd.DataFrame) -> pd.DataFrame:
    n = len(data)
    marginals = []

    for attr1, attr2 in itertools.combinations(data.columns, 2):
        marginal = (
            data.groupby([attr1, attr2], dropna=False)
            .size()
            .div(n)
            .rename("probability")
            .reset_index()
            .assign(attr1=attr1, attr2=attr2)
            .rename(columns={attr1: "value1", attr2: "value2"})
        )

        marginals.append(marginal)

    return pd.concat(marginals, ignore_index=True)[["attr1", "attr2", "value1", "value2", "probability"]]

def sample_marginals(marginals: pd.DataFrame, num_of_marginals: int, seed: int) -> pd.DataFrame:
    return marginals.sample(frac=1, random_state=seed).iloc[: num_of_marginals].reset_index(drop=True)

def add_noise(marginals: pd.DataFrame, size: int, rho: float = 1, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sigma = zcdp_gaussian_sigma(num_rows=size, rho=rho / len(marginals))
    marginals["probability"] += rng.normal(loc=0.0, scale=sigma, size=len(marginals))
    marginals["probability"] = marginals["probability"].clip(lower=0, upper=1)
    return marginals


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data = pd.read_csv(Path(cfg.input_path))
    marginals = compute_two_way_marginals(data)
    marginals = sample_marginals(marginals, cfg.num_of_marginals, cfg.seed)
    noisy_marginals = add_noise(marginals, len(data), rho=cfg.rho, seed=cfg.seed)
    noisy_marginals.to_csv(Path(cfg.output_path))


if __name__ == "__main__":
    main({
        "input_path": "resources/data/private/adult.csv",
        "output_path": "resources/marginals/adult_marginals.csv",
        "num_of_marginals": 1000,
        "rho": 1,
        "seed": 42
    })
