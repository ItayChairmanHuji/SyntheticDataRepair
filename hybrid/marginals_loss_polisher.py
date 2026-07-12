import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from denial_constraints import DenialConstraints
from graph import ConflictGraphBuilder
from marginals.marginals import Marginals
from marginals_error_calculator import MarginalsErrorCalculator
from violations_detection import find_violations

DEFAULT_BATCH_SIZE = 100
DEFAULT_ALPHA = 0.5
MIN_GAIN = 1e-10

def get_loss(
    original_size: int,
    current_size: int,
    marginals_error: float,
    alpha: float,
    has_marginals: bool = True,
) -> float:
    deleted_fraction = (original_size - current_size) / original_size if original_size > 0 else 0.0
    marginal_term = marginals_error if has_marginals else 0.0
    return alpha * deleted_fraction + (1 - alpha) * marginal_term


def phase_2(
    data: pd.DataFrame,
    marginals_error_calc: MarginalsErrorCalculator,
    batch_size: int,
    original_size: int,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:

    active_mask = np.ones(len(data), dtype=bool)
    num_deleted = 0

    while True:
        current_loss = get_loss(
            original_size,
            marginals_error_calc.size,
            marginals_error_calc.error() if marginals_error_calc.has_marginals else 0.0,
            alpha,
            marginals_error_calc.has_marginals,
        )
        active_indices = np.flatnonzero(active_mask)
        gain = get_gain(
            active_indices,
            marginals_error_calc,
            current_loss,
            original_size,
            alpha,
        )
        selected = select_tuples_to_remove(gain, active_indices, batch_size)
        selected = get_improving_selection(selected, current_loss, marginals_error_calc, original_size, alpha)
        if selected.size == 0:
            break

        active_mask[selected] = False
        num_deleted += selected.size
        marginals_error_calc.remove(selected)

    return data.iloc[np.flatnonzero(active_mask)].reset_index(drop=True)


def get_gain(
    active_indices: np.ndarray,
    marginals_error_calc: MarginalsErrorCalculator,
    current_loss: float,
    original_size: int,
    alpha: float,
) -> float | np.ndarray:
    current_size = marginals_error_calc.size
    loss_after_removal = get_loss(
        original_size,
        current_size - 1,
        marginals_error_calc.error(active_indices) if marginals_error_calc.has_marginals else 0.0,
        alpha,
        marginals_error_calc.has_marginals,
    )
    return current_loss - loss_after_removal


def select_tuples_to_remove(
    gain: np.ndarray,
    active_indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    positive = np.flatnonzero(gain > MIN_GAIN)
    if positive.size == 0:
        return np.array([], dtype=int)

    selection_size = int(min(batch_size, positive.size))
    if selection_size <= 0:
        return np.array([], dtype=int)

    if selection_size < positive.size:
        positive_gain = gain[positive]
        partitioned = np.argpartition(-positive_gain, selection_size - 1)[:selection_size]
        selected_candidates = positive[partitioned]
    else:
        selected_candidates = positive

    selected_candidates = selected_candidates[np.argsort(-gain[selected_candidates])]
    return active_indices[selected_candidates]


def get_improving_selection(
    selected: np.ndarray,
    current_loss: float,
    marginals_error_calc: MarginalsErrorCalculator,
    original_size: int,
    alpha: float,
) -> np.ndarray:
    while selected.size > 0:
        loss_after_batch = get_loss(
            original_size,
            marginals_error_calc.size - selected.size,
            marginals_error_calc.error_after_removing(selected) if marginals_error_calc.has_marginals else 0.0,
            alpha,
            marginals_error_calc.has_marginals,
        )
        if loss_after_batch < current_loss - MIN_GAIN:
            return selected
        if selected.size == 1:
            return np.array([], dtype=int)
        selected = selected[: max(1, selected.size // 2)]

    return selected

def phase_1(data: pd.DataFrame, dcs: DenialConstraints) -> pd.DataFrame:
    violations = find_violations(data, dcs)
    graph = ConflictGraphBuilder.build(len(data), violations)
    cover = []
    while graph.has_edges():
        selected = int(graph.degrees.argmax())
        cover.append(selected)
        graph.remove_vertex(selected)
    return data.drop(index=cover)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    synthetic_data = pd.read_csv(cfg.input_path)
    dcs = DenialConstraints.from_file(cfg.dcs_file)
    marginals = Marginals(Path(cfg.marginals_file))
    alpha = cfg.alpha
    tick = time.perf_counter()
    phase1 = phase_1(synthetic_data, dcs)
    error_calculator = MarginalsErrorCalculator(phase1, marginals)
    phase2 = phase_2(phase1, error_calculator, len(synthetic_data), alpha)
    tock = time.perf_counter()
    phase2.to_csv(cfg.output_path, index=False)
    print(f"Repair time: {tock-tick:.2f} seconds")
    return phase2, tock - tick

if __name__ == "__main__":
    main(DictConfig({
        "input_path": "resources/data/private/adult.csv",
        "dcs_file": "resources/dcs/adult.dcs",
        "marginals_file": "resources/marginals/adult.marginals",
        "alpha": 0.5,
        "output_path": "resources/data/repaired.csv"
    }))
