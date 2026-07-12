import time

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from denial_constraints.denial_constraints import DenialConstraints
from graph.graph import Graph
from graph.initializer import GraphBuilder
from marginals.marginals import Marginals
from marginals_error_calculator import MarginalsErrorCalculator
from marginals_loss_polisher.marginals_loss_polisher import DEFAULT_ALPHA, get_loss
from violations_detection.violations_finder import find_violations


def find_cover(
    graph: Graph,
    loss_calculator: MarginalsErrorCalculator,
    original_size: int,
    alpha: float = DEFAULT_ALPHA,
    batch_size: int = 1000,
) -> list[int]:
    cover: list[int] = []
    while graph.has_edges():
        active_indices = np.flatnonzero(graph.active)
        selected = _select_min_weighted_vertices(
            graph,
            loss_calculator,
            active_indices,
            original_size,
            alpha,
            batch_size,
        )
        removed = []
        for row_idx in selected:
            if graph.deleted[row_idx] or not graph.active[row_idx]:
                continue
            cover.append(row_idx)
            removed.append(row_idx)
            graph.remove_vertex(row_idx)
            if not graph.has_edges():
                break
        if removed:
            loss_calculator.remove(np.asarray(removed, dtype=int))
    return cover


def _select_min_weighted_vertices(
    graph: Graph,
    loss_calculator: MarginalsErrorCalculator,
    active_indices: np.ndarray,
    original_size: int,
    alpha: float,
    batch_size: int,
) -> np.ndarray:
    current_size = loss_calculator.size
    loss_after_removal = get_loss(
        original_size=original_size,
        current_size=current_size - 1,
        marginals_error=loss_calculator.error(active_indices) if loss_calculator.has_marginals else 0.0,
        alpha=alpha,
        has_marginals=loss_calculator.has_marginals,
    )
    degrees = graph.degree(active_indices)
    score = loss_after_removal / degrees
    selection_size = min(max(1, int(batch_size)), active_indices.size)
    if selection_size < active_indices.size:
        candidates = np.argpartition(score, selection_size - 1)[:selection_size]
    else:
        candidates = np.arange(active_indices.size)
    order = np.lexsort((active_indices[candidates], -degrees[candidates], score[candidates]))
    return active_indices[candidates[order]]


def repair(
    data: pd.DataFrame,
    dcs: DenialConstraints,
    marginals: Marginals,
    alpha: float = DEFAULT_ALPHA,
    original_size: int | None = None,
    batch_size: int = 1000,
) -> pd.DataFrame:
    violations_set = find_violations(data, dcs)
    graph = GraphBuilder(len(data), violations_set).graph
    loss_calculator = MarginalsErrorCalculator(data, marginals)
    cover = find_cover(graph, loss_calculator, original_size or len(data), alpha, batch_size)
    return data.drop(index=cover)


def weighted_vc_repair(
    data: pd.DataFrame,
    dcs: DenialConstraints,
    marginals: Marginals,
    alpha: float = DEFAULT_ALPHA,
    original_size: int | None = None,
    batch_size: int = 1000,
) -> tuple[pd.DataFrame, float]:
    tick = time.perf_counter()
    repaired_data = repair(data, dcs, marginals, alpha, original_size, batch_size)
    tock = time.perf_counter()
    return repaired_data, tock - tick


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    synth_data = pd.read_csv(cfg.synthetic_data_path)
    dcs = DenialConstraints(cfg.denial_constraints)
    marginals = Marginals(cfg.data_name, cfg.num_of_marginals, cfg.marginals_seed)
    return weighted_vc_repair(
        synth_data,
        dcs,
        marginals,
        alpha=cfg.get("alpha", DEFAULT_ALPHA),
        original_size=cfg.get("original_size", len(synth_data)),
        batch_size=cfg.get("batch_size", 1000),
    )


if __name__ == "__main__":
    main()
