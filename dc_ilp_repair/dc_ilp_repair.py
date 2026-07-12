import os
import time

import gurobipy as gp
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from denial_constraints.denial_constraints import DenialConstraints
from violations_detection.models import Violation, ViolationSet
from violations_detection.violations_finder import find_violations


def repair(data: pd.DataFrame, dcs: DenialConstraints) -> pd.DataFrame:
    violations_set = find_violations(data, dcs)
    if not violations_set:
        return data.copy()

    deletion_counts = _solve_min_deletions(violations_set)
    deletion_indices = _expand_deletion_counts(violations_set, deletion_counts)
    return data.drop(index=data.index[deletion_indices])


def _solve_min_deletions(violations_set: ViolationSet) -> np.ndarray:
    cluster_sizes = violations_set.cluster_sizes
    os.environ["GRB_LICENSE_FILE"] = "./gurobi.lic"
    model = gp.Model("dc_ilp_repair")
    model.Params.TimeLimit = 3600 * 4
    model.Params.NodefileStart = 0.5
    model.Params.PreSparsify = 1

    delete = model.addVars(
        violations_set.num_clusters,
        vtype=gp.GRB.INTEGER,
        name="delete",
    )
    for cluster_id, cluster_size in enumerate(cluster_sizes):
        delete[cluster_id].LB = 0
        delete[cluster_id].UB = int(cluster_size)

    model.setObjective(
        gp.quicksum(delete[cluster_id] for cluster_id in range(violations_set.num_clusters)),
        gp.GRB.MINIMIZE,
    )
    for block_id, violation in enumerate(violations_set.violations):
        _add_violation_constraints(model, delete, cluster_sizes, block_id, violation)

    model.optimize()

    if model.SolCount == 0:
        raise RuntimeError(f"Gurobi did not find a feasible repair. Solver status: {model.Status}")

    return np.asarray(
        [int(round(delete[cluster_id].X)) for cluster_id in range(violations_set.num_clusters)],
        dtype=int,
    )


def _add_violation_constraints(
    model: gp.Model,
    delete: gp.tupledict,
    cluster_sizes: np.ndarray,
    block_id: int,
    violation: Violation,
) -> None:
    left = _unique_cluster_ids(violation.left)
    right = left if violation.symmetric else _unique_cluster_ids(violation.right)
    if len(left) == 0 or len(right) == 0:
        return

    left_set = set(left.tolist())
    right_set = set(right.tolist())
    union = _unique_cluster_ids(np.concatenate((left, right)))

    if violation.symmetric or left_set == right_set:
        _add_clique_constraint(model, delete, cluster_sizes, union, f"clique[{block_id}]")
        return

    left_kept = _kept_expr(delete, cluster_sizes, left)
    right_kept = _kept_expr(delete, cluster_sizes, right)
    left_total = _cluster_total(cluster_sizes, left)
    right_total = _cluster_total(cluster_sizes, right)
    if left_total == 0 or right_total == 0:
        return

    if left_set.isdisjoint(right_set):
        keep_left = model.addVar(vtype=gp.GRB.BINARY, name=f"side[{block_id}]")
        model.addConstr(left_kept <= left_total * keep_left, name=f"left_side[{block_id}]")
        model.addConstr(right_kept <= right_total * (1 - keep_left), name=f"right_side[{block_id}]")
        return

    keep_left_only = model.addVar(vtype=gp.GRB.BINARY, name=f"left_only[{block_id}]")
    keep_right_only = model.addVar(vtype=gp.GRB.BINARY, name=f"right_only[{block_id}]")
    keep_single = model.addVar(vtype=gp.GRB.BINARY, name=f"single[{block_id}]")
    union_kept = _kept_expr(delete, cluster_sizes, union)
    union_total = _cluster_total(cluster_sizes, union)

    model.addConstr(
        keep_left_only + keep_right_only + keep_single == 1,
        name=f"overlap_mode[{block_id}]",
    )
    model.addConstr(
        right_kept <= right_total * (1 - keep_left_only),
        name=f"overlap_delete_right[{block_id}]",
    )
    model.addConstr(
        left_kept <= left_total * (1 - keep_right_only),
        name=f"overlap_delete_left[{block_id}]",
    )
    model.addConstr(
        union_kept <= 1 + max(union_total - 1, 0) * (1 - keep_single),
        name=f"overlap_single[{block_id}]",
    )


def _add_clique_constraint(
    model: gp.Model,
    delete: gp.tupledict,
    cluster_sizes: np.ndarray,
    cluster_ids: np.ndarray,
    name: str,
) -> None:
    total = _cluster_total(cluster_sizes, cluster_ids)
    if total > 1:
        model.addConstr(_kept_expr(delete, cluster_sizes, cluster_ids) <= 1, name=name)


def _kept_expr(delete: gp.tupledict, cluster_sizes: np.ndarray, cluster_ids: np.ndarray):
    return gp.quicksum(int(cluster_sizes[cluster_id]) - delete[int(cluster_id)] for cluster_id in cluster_ids)


def _cluster_total(cluster_sizes: np.ndarray, cluster_ids: np.ndarray) -> int:
    return int(cluster_sizes[cluster_ids].sum())


def _unique_cluster_ids(cluster_ids: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(cluster_ids, dtype=int))


def _expand_deletion_counts(violations_set: ViolationSet, deletion_counts: np.ndarray) -> list[int]:
    deletion_indices: list[int] = []
    for cluster_id, delete_count in enumerate(deletion_counts):
        if delete_count <= 0:
            continue
        cluster_rows = violations_set.cluster_indices[cluster_id]
        deletion_indices.extend(int(row_idx) for row_idx in cluster_rows[:delete_count])
    return deletion_indices


def dc_ilp_repair(data: pd.DataFrame, dcs: DenialConstraints):
    tick = time.perf_counter()
    repaired_data = repair(data, dcs)
    tock = time.perf_counter()
    return repaired_data, tock - tick


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    synth_data = pd.read_csv(cfg.input_path)
    dcs = DenialConstraints.from_file(cfg.dcs_file)
    output = dc_ilp_repair(synth_data, dcs)
    output[0].to_csv(cfg.output_path, index=False)
    print(f"Repair time: {output[1]:.2f} seconds")
    return output

if __name__ == "__main__":
    main(DictConfig({
        "input_path": "resources/data/data.csv",
        "dcs_file": "resources/constraints/adult.txt",
        "output_path": "resources/data/repaired.csv"
    }))
