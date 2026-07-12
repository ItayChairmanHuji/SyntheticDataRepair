import time
from pathlib import Path

import dill
import hydra
import pandas as pd
from omegaconf import DictConfig
from snsynth import Synthesizer


def train_model(data: pd.DataFrame, model: str, epsilon: float) -> tuple[Synthesizer, float]:
    synth = Synthesizer.create(model, epsilon=epsilon, verbose=True)
    start_time = time.perf_counter()
    synth.fit(data, categorical_columns=data.columns)
    training_time = time.perf_counter() - start_time
    return synth, training_time

def save_model(model: Synthesizer, output_path: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        dill.dump(model, f)


def save_runtime(training_time: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.with_stem(f"{output_path.stem}_runtime").with_suffix(".txt")
    with open(output_path, "w") as f:
        f.write(f"Training time: {training_time:.2f} seconds\n")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data = pd.read_csv(Path(cfg.input_path))
    synth, training_time = train_model(data, cfg.model, cfg.epsilon)
    save_model(synth, Path(cfg.input_path))
    save_runtime(training_time, Path(cfg.output_dir))


if __name__ == "__main__":
    main(DictConfig({
        "input_path": "resources/data/private/adult.csv",
        "output_dir": "resources/models/adult_mst_1.0.dill",
        "model": "mst",
        "epsilon": 1.0
    }))
