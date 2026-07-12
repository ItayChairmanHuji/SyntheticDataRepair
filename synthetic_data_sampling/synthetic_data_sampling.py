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


def load_model(model_path: Path) -> Synthesizer:
    with open(model_path, "rb") as f:
        model = dill.load(f)
    return model

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.synthetic_seed)
    model = load_model(Path(cfg.model_path))
    data = model.sample(size)
    data.to_csv(Path(cfg.output_path))

if __name__ == "__main__":
    main(DictConfig({
        "model_path": "resources/models/adult_mst_1.0.dill",
        "output_path": "resources/data/synthetic/adult_synthetic.csv",
        "synthetic_seed": 42,
        "size": 10000
    }))
