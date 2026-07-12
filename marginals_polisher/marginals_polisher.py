import uuid
from dataclasses import replace

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from dataset.dataset import Dataset
from marginals_error_calculator import MarginalsErrorCalculator

DEFAULT_BATCH_SIZE = 100
MIN_GAIN = 1e-10


def polish(
    data: pd.DataFrame,
    marginals_error_calc: MarginalsErrorCalculator,
    max_deletions: int,
    batch_size: int,
    min_gain: float = MIN_GAIN,
) -> pd.DataFrame:
    max_deletions = min(max_deletions, max(len(data) - 1, 0))
    if max_deletions == 0 or len(data) <= 1 or not marginals_error_calc.has_marginals:
        return data.copy()

    active_mask = np.ones(len(data), dtype=bool)
    num_deleted = 0

    while num_deleted < max_deletions and marginals_error_calc.size > 1:
        current_error = marginals_error_calc.error()
        active_indices = np.flatnonzero(active_mask)
        gain = get_gain(active_indices, marginals_error_calc, current_error)
        selected = select_tuples_to_remove(gain, active_indices, max_deletions, num_deleted, batch_size, min_gain)
        selected = get_improving_selection(selected, current_error, marginals_error_calc, min_gain)
        if selected.size == 0:
            break

        active_mask[selected] = False
        num_deleted += selected.size
        marginals_error_calc.remove(selected)

    return data.iloc[np.flatnonzero(active_mask)].reset_index(drop=True)


def get_gain(
    active_indices: np.ndarray,
    marginals_error_calc: MarginalsErrorCalculator,
    current_error: float | np.ndarray,
) -> float | np.ndarray:
    error_on_removal = marginals_error_calc.error(active_indices)
    gain = current_error - error_on_removal
    return gain


def select_tuples_to_remove(
    gain: np.ndarray,
    active_indices: np.ndarray,
    max_deletions: int,
    num_deleted: int,
    batch_size: int,
    min_gain: float = MIN_GAIN,
) -> np.ndarray:
    positive = np.flatnonzero(gain > min_gain)
    if positive.size == 0:
        return np.array([], dtype=int)

    remaining_budget = max_deletions - num_deleted
    selection_size = int(min(batch_size, remaining_budget, positive.size))
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
    current_error: float | np.ndarray,
    marginals_error_calc: MarginalsErrorCalculator,
    min_gain: float = MIN_GAIN,
) -> np.ndarray:
    while selected.size > 0:
        error_after_batch = marginals_error_calc.error_after_removing(selected)
        if error_after_batch < current_error - min_gain:
            return selected
        if selected.size == 1:
            return np.array([], dtype=int)
        selected = selected[: max(1, selected.size // 2)]

    return selected


def get_max_deletions(data: Dataset, deletion_budget: float) -> int:
    if deletion_budget < 0 or deletion_budget > 1:
        raise ValueError("deletion_budget must be between 0 and 1.")

    return min(int(deletion_budget * len(data)), max(len(data) - 1, 0))


def generate_repaired_id(synth_data: Dataset) -> str:
    return f"{synth_data.metadata.dataset_name}_{synth_data.metadata.sweep_name}_marginals_polish_{uuid.uuid4()}"


def create_metadata(data: Dataset, id: str):
    return replace(
        data.metadata,
        dataset_id=id,
        repair_algorithm="marginals_polish",
    )


def marginals_polisher(
    data: Dataset,
    deletion_budget: float,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_gain: float = MIN_GAIN,
) -> Dataset:
    polished_id = generate_repaired_id(data)
    marginals_error_calc = MarginalsErrorCalculator(data)
    max_deletions = get_max_deletions(data, deletion_budget)
    polished_data = polish(data.data, marginals_error_calc, max_deletions, batch_size, min_gain)
    metadata = create_metadata(data, polished_id)
    Dataset.save_dataset(polished_data, metadata, polished_id)
    return Dataset(polished_id)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    synth_data = Dataset(cfg.synth_data_id)
    return marginals_polisher(synth_data, cfg.deletion_budget, cfg.get("batch_size", DEFAULT_BATCH_SIZE))


if __name__ == "__main__":
    main()
