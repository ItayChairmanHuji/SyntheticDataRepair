import random
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from snsynth import Synthesizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(data_name: str, model_name: str, epsilon: float) -> Synthesizer:
    input_dir = Path("resources/models/")
    input_path = input_dir / f"{data_name}_{model_name}_{epsilon}.csv"
    with open(input_path, "rb") as f:
        model = dill.load(f)
    return model


def synthetic_data_sampling(synthetic_seed: int, dataset_name: str, model_name: str, epsilon: float, size: int):
    set_seed(synthetic_seed)
    model = load_model(dataset_name, model_name, epsilon)
    return model.sample(size)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    synthetic_data_sampling(
        synthetic_seed=cfg.synthetic_seed,
        dataset_name=cfg.dataset,
        model_name=cfg.model,
        epsilon=cfg.epsilon,
        size=cfg.size,
    )


if __name__ == "__main__":
    main()
