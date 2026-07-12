import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gurobipy as gp
import numpy as np
import pandas as pd
from omegaconf import ListConfig

from dc_ilp_repair.dc_ilp_repair import _expand_deletion_counts
from denial_constraints import DenialConstraint
from denial_constraints.denial_constraints import DenialConstraints
from evaluation.tvd import average_two_way_tvd
from marginals.marginal import Marginal
from marginals.marginals import Marginals
from violations_detection import ViolationFinder, ViolationSet, compact_data

DEFAULT_ALPHA = 0.5
MIN_TARGET = 1e-8
DEFAULT_MARGINALS_SIZE = 50
DEFAULT_TIME_LIMIT_SECONDS = 3600 * 4


@dataclass(frozen=True)
class RepairProblem:
    violations_set: ViolationSet
    compact_df: pd.DataFrame


def repair(
    data: pd.DataFrame,
    dcs: DenialConstraints,
    marginals: Iterable[Marginal],
    alpha: float = DEFAULT_ALPHA,
    time_limit: int = DEFAULT_TIME_LIMIT_SECONDS,
    threads: int | None = None,
) -> pd.DataFrame:
    marginals = list(marginals)
    problem = _build_problem(data, dcs, marginals)
    deletion_counts = _solve_loss_repair(
        problem,
        marginals,
        data_size=len(data),
        alpha=alpha,
        time_limit=time_limit,
        threads=threads,
    )
    deletion_indices = _expand_deletion_counts(problem.violations_set, deletion_counts)
    return data.drop(index=data.index[deletion_indices])


def _build_problem(
    data: pd.DataFrame,
    dcs: DenialConstraints,
    marginals: list[Marginal],
) -> RepairProblem:
    attrs = sorted(set(dcs.attrs).union(*(marginal.attrs for marginal in marginals)))
    compact = compact_data(data, attrs)
    violations_set = ViolationFinder().find_compact_violations(compact, dcs.constraints)
    return RepairProblem(violations_set=violations_set, compact_df=compact.df)


def _solve_loss_repair(
    problem: RepairProblem,
    marginals: list[Marginal],
    data_size: int,
    alpha: float = DEFAULT_ALPHA,
    time_limit: int = DEFAULT_TIME_LIMIT_SECONDS,
    threads: int | None = None,
) -> np.ndarray:
    violations_set = problem.violations_set
    cluster_sizes = violations_set.cluster_sizes
    os.environ["GRB_LICENSE_FILE"] = "./gurobi.lic"
    model = gp.Model("loss_ilp_repair")
    model.Params.TimeLimit = time_limit
    model.Params.NodefileStart = 0.5
    model.Params.PreSparsify = 1
    model.Params.LazyConstraints = 1
    if threads is not None:
        model.Params.Threads = threads

    delete = model.addVars(
        violations_set.num_clusters,
        vtype=gp.GRB.INTEGER,
        name="delete",
    )
    for cluster_id, cluster_size in enumerate(cluster_sizes):
        delete[cluster_id].LB = 0
        delete[cluster_id].UB = int(cluster_size)

    keep = {
        cluster_id: int(cluster_sizes[cluster_id]) - delete[cluster_id]
        for cluster_id in range(violations_set.num_clusters)
    }
    kept_total = gp.quicksum(keep[cluster_id] for cluster_id in range(violations_set.num_clusters))
    _add_binary_expanded_total_kept(model, kept_total, data_size)
    deletion_term = alpha * (data_size - kept_total) / data_size
    objective = deletion_term

    model.addConstr(kept_total >= 1, name="keep_at_least_one")
    if marginals:
        objective += _add_marginal_objective(model, keep, problem, marginals, kept_total, alpha)
    active = _add_active_cluster_variables(model, keep, cluster_sizes)

    model.setObjective(objective, gp.GRB.MINIMIZE)
    model.optimize(_lazy_conflict_callback(active, violations_set))

    if model.SolCount == 0:
        return np.asarray(
            [int(delete[cluster_id].UB) for cluster_id in range(violations_set.num_clusters)],
            dtype=int,
        )

    return np.asarray(
        [int(round(delete[cluster_id].X)) for cluster_id in range(violations_set.num_clusters)],
        dtype=int,
    )


def _add_active_cluster_variables(
    model: gp.Model,
    keep: dict[int, gp.LinExpr],
    cluster_sizes: np.ndarray,
) -> gp.tupledict:
    active = model.addVars(len(cluster_sizes), vtype=gp.GRB.BINARY, name="active")
    for cluster_id, cluster_size in enumerate(cluster_sizes):
        model.addConstr(keep[cluster_id] <= int(cluster_size) * active[cluster_id], name=f"active_ub[{cluster_id}]")
        model.addConstr(keep[cluster_id] >= active[cluster_id], name=f"active_lb[{cluster_id}]")
    return active


def _lazy_conflict_callback(active: gp.tupledict, violations_set: ViolationSet):
    prepared = [_prepare_violation(violation) for violation in violations_set.violations]

    def callback(model: gp.Model, where: int) -> None:
        if where != gp.GRB.Callback.MIPSOL:
            return

        active_values = model.cbGetSolution(active)
        is_active = np.fromiter(
            (active_values[cluster_id] > 0.5 for cluster_id in range(violations_set.num_clusters)),
            dtype=bool,
            count=violations_set.num_clusters,
        )

        for left, right, symmetric in prepared:
            if symmetric:
                kept = left[is_active[left]]
                for first, second in combinations(kept.tolist(), 2):
                    model.cbLazy(active[int(first)] + active[int(second)] <= 1)
                continue

            active_left = left[is_active[left]]
            if len(active_left) == 0:
                continue
            active_right = right[is_active[right]]
            if len(active_right) == 0:
                continue

            for left_id in active_left:
                for right_id in active_right:
                    if int(left_id) != int(right_id):
                        model.cbLazy(active[int(left_id)] + active[int(right_id)] <= 1)

    return callback


def _prepare_violation(violation) -> tuple[np.ndarray, np.ndarray, bool]:
    left = np.unique(np.asarray(violation.left, dtype=int))
    if violation.symmetric:
        return left, left, True
    right = np.unique(np.asarray(violation.right, dtype=int))
    return left, right, False


def _add_marginal_objective(
    model: gp.Model,
    keep: dict[int, gp.LinExpr],
    problem: RepairProblem,
    marginals: list[Marginal],
    kept_total,
    alpha: float,
):
    violations_set = problem.violations_set
    marginal_error_terms = []
    kept_total_units = getattr(model, "_loss_ilp_kept_total_units")

    for marginal_id, marginal in enumerate(marginals):
        distance = model.addVar(
            lb=0.0,
            ub=1.0,
            vtype=gp.GRB.CONTINUOUS,
            name=f"marginal_distance[{marginal_id}]",
        )
        matching_clusters = np.flatnonzero(marginal.mask(problem.compact_df).to_numpy(dtype=bool))
        kept_matching = gp.quicksum(
            keep[int(cluster_id)] for cluster_id in matching_clusters
        )
        target = float(marginal.target)
        weighted_z_sum = gp.LinExpr()
        for unit_id, unit in enumerate(kept_total_units):
            z = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=gp.GRB.CONTINUOUS,
                name=f"z[{marginal_id},{unit_id}]",
            )
            model.addConstr(z <= distance, name=f"z_distance_ub[{marginal_id},{unit_id}]")
            model.addConstr(z <= unit["var"], name=f"z_keep_ub[{marginal_id},{unit_id}]")
            model.addConstr(
                z >= distance - (1 - unit["var"]),
                name=f"z_lb[{marginal_id},{unit_id}]",
            )
            weighted_z_sum += int(unit["weight"]) * z

        model.addConstr(
            weighted_z_sum >= kept_matching - target * kept_total,
            name=f"marginal_pos[{marginal_id}]",
        )
        model.addConstr(
            weighted_z_sum >= target * kept_total - kept_matching,
            name=f"marginal_neg[{marginal_id}]",
        )
        model.addConstr(
            distance >= target * (1 - kept_total),
            name=f"marginal_empty_lower_bound[{marginal_id}]",
        )

        marginal_error_terms.append(distance / max(abs(target), MIN_TARGET))

    return (1 - alpha) * gp.quicksum(marginal_error_terms) / len(marginals)


def _add_binary_expanded_total_kept(
    model: gp.Model,
    kept_total,
    data_size: int,
) -> None:
    kept_total_units = []
    kept_total_expr = gp.LinExpr()
    for unit_offset, weight in enumerate(_binary_weights(data_size)):
        unit = model.addVar(vtype=gp.GRB.BINARY, name=f"kept_total_unit[{unit_offset}]")
        kept_total_units.append({"weight": int(weight), "var": unit})
        kept_total_expr += int(weight) * unit

    model.addConstr(kept_total_expr == kept_total, name="kept_total_binary_expansion")
    model._loss_ilp_kept_total_units = kept_total_units


def _binary_weights(total: int) -> list[int]:
    weights: list[int] = []
    next_weight = 1
    remaining = total
    while remaining > 0:
        weight = min(next_weight, remaining)
        weights.append(weight)
        remaining -= weight
        next_weight *= 2
    return weights


def _load_denial_constraints(config_value) -> DenialConstraints:
    if isinstance(config_value, (list, ListConfig)):
        constraints = [DenialConstraint.from_string(raw) if isinstance(raw, str) else raw for raw in config_value]
        return DenialConstraints(constraints)

    path_or_name = Path(str(config_value))
    if path_or_name.exists() or path_or_name.suffix:
        return DenialConstraints.from_file(path_or_name)
    return DenialConstraints.from_dataset_name(path_or_name)


def loss_ilp_repair(
    data: pd.DataFrame,
    dcs: DenialConstraints,
    marginals: Marginals,
    alpha: float = DEFAULT_ALPHA,
    normalize_by_repaired_size: bool = True,
    time_limit: int = DEFAULT_TIME_LIMIT_SECONDS,
    threads: int | None = None,
):
    tick = time.perf_counter()
    repaired_data = repair(data, dcs, marginals, alpha, time_limit=time_limit, threads=threads)
    tock = time.perf_counter()
    return repaired_data, tock - tick


def _repair_file(args: argparse.Namespace, input_path: Path, output_dir: Path, source_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{source_name}] reading {input_path}")
    data = pd.read_csv(input_path)
    if args.rows is not None:
        data = data.head(args.rows).copy()

    dcs = _load_denial_constraints(args.denial_constraints or args.data_name)
    marginals = Marginals(args.data_name, args.marginals_size, args.marginals_seed)
    marginals.filter(args.marginals_filter_size or len(marginals))

    tick = time.perf_counter()
    repaired = repair(data, dcs, marginals, alpha=args.alpha, time_limit=args.time_limit, threads=args.threads)
    runtime = time.perf_counter() - tick

    repaired.to_csv(output_dir / "data.csv", index=False)
    results = {
        "source": source_name,
        "input_path": str(input_path),
        "rows_in": int(len(data)),
        "rows_out": int(len(repaired)),
        "deletion_ratio": float(1 - len(repaired) / max(len(data), 1)),
        "runtime": runtime,
        "alpha": args.alpha,
        "marginals_size": args.marginals_size,
        "marginals_seed": args.marginals_seed,
        "marginals_error": marginals.error(repaired),
        "marginals_distance": marginals.distance(repaired),
    }
    private_path = Path(f"resources/data/private/{args.data_name}/data.csv")
    if private_path.exists():
        private = pd.read_csv(private_path)
        results["marginals_error_private"] = marginals.error(repaired, private)
        results["marginals_distance_private"] = marginals.distance(repaired, private)
        results["tvd"] = average_two_way_tvd(repaired, private)

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"[{source_name}] wrote {output_dir / 'data.csv'} in {runtime:.2f}s")


def _resolve_input_path(args: argparse.Namespace, source_name: str) -> Path:
    if args.input_path is not None:
        return args.input_path
    if args.experiment_name == "size_sweep":
        input_size = args.input_size or args.rows
        return (
            Path("resources/data/size_sweep")
            / source_name
            / f"{args.data_name}_{args.model}_{input_size}_{args.synthetic_seed}"
            / "data.csv"
        )
    return (
        Path("resources/data/epsilon_sweep")
        / source_name
        / f"{args.data_name}_{args.model}_{args.epsilon}_{args.synthetic_seed}"
        / "data.csv"
    )


def _default_output_dir(args: argparse.Namespace, source_name: str) -> Path:
    input_size = args.input_size or args.rows
    suffix = (
        f"{args.data_name}_{args.model}_{input_size}_{args.synthetic_seed}_{args.marginals_seed}"
        if args.experiment_name == "size_sweep"
        else f"{args.data_name}_{args.model}_{args.epsilon}_{args.synthetic_seed}_{args.marginals_seed}"
    )
    return (
        args.output_base
        / args.experiment_name
        / source_name
        / suffix
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run loss-aware ILP repair from loss_ilp_repair/ilp_details.md.")
    parser.add_argument("--input-path", type=Path, default=None, help="CSV to repair. If omitted, use epsilon_sweep layout.")
    parser.add_argument("--output-base", type=Path, default=Path("resources/data/loss_ilp_repair"))
    parser.add_argument("--experiment-name", choices=["size_sweep", "epsilon_sweep"], default="size_sweep")
    parser.add_argument("--source", choices=["synth", "vanilla", "both"], default="both")
    parser.add_argument("--data-name", default="adult")
    parser.add_argument("--model", default="mst")
    parser.add_argument("--epsilon", default="0.5")
    parser.add_argument("--synthetic-seed", type=int, default=0)
    parser.add_argument("--input-size", type=int, default=None, help="Size token used in size_sweep input/output paths.")
    parser.add_argument("--rows", type=int, default=500_000, help="Maximum rows to load from the input CSV.")
    parser.add_argument("--denial-constraints", default=None)
    parser.add_argument("--marginals-size", type=int, default=DEFAULT_MARGINALS_SIZE)
    parser.add_argument("--marginals-seed", type=int, default=0)
    parser.add_argument("--marginals-filter-size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT_SECONDS)
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sources = ["synth", "vanilla"] if args.source == "both" else [args.source]
    for source_name in sources:
        input_path = _resolve_input_path(args, source_name)
        if not input_path.exists():
            raise FileNotFoundError(f"Could not find input for {source_name}: {input_path}")
        output_dir = _default_output_dir(args, source_name)
        _repair_file(args, input_path, output_dir, source_name)


if __name__ == "__main__":
    main()
