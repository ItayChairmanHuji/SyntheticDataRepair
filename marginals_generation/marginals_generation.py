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


def add_noise(marginals: pd.DataFrame, size: int, rho: float = 1, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sigma = zcdp_gaussian_sigma(num_rows=size, rho=rho / len(marginals))
    marginals["probability"] += rng.normal(loc=0.0, scale=sigma, size=len(marginals))
    marginals["probability"] = marginals["probability"].clip(lower=0, upper=1)
    return marginals


def load_input(data_name: str, input_dir: Path) -> pd.DataFrame:
    input_path = input_dir / f"{data_name}_private" / "data.csv"
    return pd.read_csv(input_path)


def save_output(marginals: pd.DataFrame, data_name: str, seed: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{data_name}_{len(marginals)}_{seed}.csv"
    marginals.to_csv(output_path, index=False)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data = load_input(cfg.data_name, Path(cfg.input_dir))
    marginals = compute_two_way_marginals(data)
    marginals = marginals.sample(frac=1, random_state=cfg.seed).iloc[: cfg.num_of_marginals].reset_index(drop=True)
    noisy_marginals = add_noise(marginals, len(data), rho=cfg.rho, seed=cfg.seed)
    save_output(noisy_marginals, cfg.data_name, cfg.seed, Path(cfg.output_dir))


if __name__ == "__main__":
    main()
