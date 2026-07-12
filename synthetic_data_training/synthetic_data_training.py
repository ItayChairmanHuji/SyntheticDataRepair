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


def load_input(data_name: str, input_dir: Path) -> pd.DataFrame:
    input_path = input_dir / f"{data_name}_private" / "data.csv"
    return pd.read_csv(input_path)


def save_model(model: Synthesizer, data_name: str, output_dir: Path, eps: float, model_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{data_name}_{model_name}_{eps}.csv"
    with open(output_path, "wb") as f:
        dill.dump(model, f)


def save_runtime(training_time: float, data_name: str, output_dir: Path, eps: float, model_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{data_name}_{model_name}_{eps}_runtime.txt"
    with open(output_path, "w") as f:
        f.write(f"Training time: {training_time:.2f} seconds\n")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data = load_input(cfg.data_name, Path(cfg.input_dir))
    synth, training_time = train_model(data, cfg.model, cfg.epsilon)
    save_model(synth, cfg.data_name, Path(cfg.output_dir), cfg.epsilon, cfg.model)
    save_runtime(training_time, cfg.data_name, Path(cfg.output_dir), cfg.epsilon, cfg.model)


if __name__ == "__main__":
    main()
