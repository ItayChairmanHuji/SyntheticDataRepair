import time
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from archive.p_processes.p04_repairing.p04a_vanilla_repairing.src.core.vanilla_vc_repairer import VanillaVCRepairer
from archive.p_processes.p04_repairing.src.core import Repairer
from archive.u_utilities.u_shared import Dataset, MarginalSet


@dataclass
class VanillaMarginalPolishRepairer(Repairer):
    extra_delete_budget: float = 0.05
    batch_size: int = 100
    min_gain: float = 1e-10
    alpha: float = 0.5
    profiler: Dict[str, Any] = field(default_factory=dict)

    def repair(self, dataset: Dataset, marginals: MarginalSet) -> Dataset:
        t_start = time.perf_counter()

        vanilla = VanillaVCRepairer(alpha=self.alpha)
        vanilla_repaired = vanilla.repair(dataset, marginals)
        vanilla_repaired.runtimes.update(dataset.runtimes)

        self.profiler["vanilla"] = vanilla.profiler.copy()
        self.profiler["original_n"] = len(dataset.data)
        self.profiler["vanilla_n"] = len(vanilla_repaired.data)
        self.profiler["vanilla_deleted"] = len(dataset.data) - len(vanilla_repaired.data)

        t_polish = time.perf_counter()
        polished = self._polish(vanilla_repaired, marginals, len(dataset.data))
        self.profiler["polish_s"] = time.perf_counter() - t_polish
        self.profiler["repair_total_s"] = time.perf_counter() - t_start

        polished.profiling_stats = self.profiler.copy()
        return polished

    def _polish(self, dataset: Dataset, marginals: MarginalSet, original_n: int) -> Dataset:
        n = len(dataset.data)
        max_extra_deleted = int(original_n * self.extra_delete_budget)

        self.profiler["extra_delete_budget"] = self.extra_delete_budget
        self.profiler["max_extra_deleted"] = max_extra_deleted

        if n <= 1 or len(marginals) == 0 or max_extra_deleted <= 0:
            self.profiler["extra_deleted"] = 0
            self.profiler["stop_reason"] = "no_budget_or_marginals"
            return dataset

        matching_matrix = self._build_matching_matrix(dataset, marginals)
        target_freqs = np.array([m.target for m in marginals], dtype=float)
        safe_targets = np.maximum(target_freqs, 1e-5)

        active = np.ones(n, dtype=bool)
        current_counts = matching_matrix.sum(axis=0).astype(float)
        current_n = n
        current_error = self._avg_relative_error(current_counts, current_n, target_freqs, safe_targets)
        initial_error = current_error

        extra_deleted = 0
        iterations = 0
        best_gain_seen = 0.0
        stop_reason = "budget_exhausted"

        while extra_deleted < max_extra_deleted and current_n > 1:
            active_indices = np.flatnonzero(active)
            if active_indices.size == 0:
                stop_reason = "empty_dataset"
                break

            active_matrix = matching_matrix[active_indices].astype(float, copy=False)
            new_n = current_n - 1
            candidate_counts = current_counts[None, :] - active_matrix
            candidate_errors = np.abs(candidate_counts / new_n - target_freqs) / safe_targets
            gains = current_error - candidate_errors.mean(axis=1)

            positive = np.flatnonzero(gains > self.min_gain)
            if positive.size == 0:
                stop_reason = "no_positive_gain"
                break

            remaining_budget = max_extra_deleted - extra_deleted
            take = min(self.batch_size, remaining_budget, positive.size)
            if take < positive.size:
                positive_gains = gains[positive]
                chosen_local = positive[np.argpartition(-positive_gains, take - 1)[:take]]
                chosen_local = chosen_local[np.argsort(-gains[chosen_local])]
            else:
                chosen_local = positive[np.argsort(-gains[positive])]

            chosen_indices = active_indices[chosen_local]
            best_gain_seen = max(best_gain_seen, float(gains[chosen_local[0]]))

            active[chosen_indices] = False
            current_counts -= matching_matrix[chosen_indices].sum(axis=0)
            current_n -= len(chosen_indices)
            extra_deleted += len(chosen_indices)
            iterations += 1
            current_error = self._avg_relative_error(current_counts, current_n, target_freqs, safe_targets)

        keep = np.flatnonzero(active)
        repaired = Dataset(
            name=f"{dataset.name}_repaired",
            data=dataset.data.iloc[keep].reset_index(drop=True),
            dcs=dataset.dcs,
            target=dataset.target,
            mappings=dataset.mappings,
        )

        self.profiler["polish_initial_marginal_error"] = float(initial_error)
        self.profiler["polish_final_marginal_error"] = float(current_error)
        self.profiler["polish_best_gain_seen"] = float(best_gain_seen)
        self.profiler["extra_deleted"] = extra_deleted
        self.profiler["polish_iterations"] = iterations
        self.profiler["stop_reason"] = stop_reason
        self.profiler["final_n"] = len(repaired.data)
        return repaired

    def _build_matching_matrix(self, dataset: Dataset, marginals: MarginalSet) -> np.ndarray:
        matrix = np.zeros((len(dataset.data), len(marginals)), dtype=bool)
        for i, marginal in enumerate(marginals):
            matrix[:, i] = marginal.get_mask(dataset.data, dataset.mappings)
        return matrix

    def _avg_relative_error(
        self,
        counts: np.ndarray,
        n: int,
        target_freqs: np.ndarray,
        safe_targets: np.ndarray,
    ) -> float:
        return float(np.mean(np.abs(counts / n - target_freqs) / safe_targets))
