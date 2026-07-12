# Violation Detection

`ViolationFinder` is the public facade. It accepts a dataset object with `dcs` and `compact()` attributes.

The finder compacts rows first, then emits symbolic `Violation` blocks between compact clusters. This avoids materializing every row-pair edge during detection.

## Module Map

- `violations_finder.py`: public facade and route dispatch.
- `constraint_plan.py`: classifies one denial constraint into a simple plan.
- `fast_paths.py`: optimized implementations for the common constraint shapes.
- `frames.py`: pandas grouping/filtering helpers.
- `predicate_eval.py`: value comparison and tuple-filter evaluation.
- `conflicts.py`: small helpers that append symbolic violation blocks.
- `models.py`: compact data and violation data structures.
- `duckdb_engine.py`: lazy fallback for complex constraints.

## Fast Paths

The common denial-constraint shapes stay in pandas/NumPy:

- Equality plus one not-equal predicate: group by equality attributes, then connect groups with different values.
- Optional tuple filters: apply `t1` and `t2` filters before grouping.
- One or two order predicates: group by equality attributes when present, then use sorted numeric arrays and `np.searchsorted`.
- Equality/filter-only constraints: connect all matching clusters directly.

These cover the constraints in `resources/constraints/*.txt`, including grouped order constraints such as:

```text
not(t1.State=t2.State&t1.Salary>t2.Salary&t1.Rate<t2.Rate)
```

## Fallback

Complex predicates that do not match those shapes use `DuckDBEngine`. DuckDB is imported lazily, so the package can be imported and the fast paths can run without DuckDB installed. If a complex constraint needs the fallback and DuckDB is missing, the finder raises an `ImportError` with installation guidance.
