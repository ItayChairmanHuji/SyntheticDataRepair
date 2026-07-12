# Conflict Graph Implementation

This directory has two graph representations for the same conflict structure:

- `Graph`, the row-level compressed graph used by default.
- `ClusterCountGraph`, the lower-memory graph used for vanilla repair on `adult` and `tax` PATE-CTGAN runs.

Both start from the same idea: the violation detector does not return individual edges. It returns compact conflicts between clusters of identical rows. A conflict says, in effect:

> Every row in these left clusters conflicts with every row in these right clusters.

So the graph is not stored as an edge list. It is stored as block pairs.

## Basic Objects

### Cluster

A cluster is a group of row indices that have the same values over the denial-constraint attributes.

`ClusterMap` stores:

- `row_to_cluster`: array of size `n`, mapping each row index to its cluster id.
- `members`: tuple where `members[cid]` is the array of row indices in cluster `cid`.

### Block Pair

A block pair is a symbolic conflict between two sets of clusters:

- `left_clusters`
- `right_clusters`

If the conflict is symmetric, the left and right side are the same and the block represents a clique over that block.

The row-level `BlockPair` also precomputes:

- `left_members`
- `right_members`
- `union_members`

These arrays are the concrete row indices represented by the cluster sets.

### Membership Side

When a cluster participates in a block pair, the graph records which side it is on:

- `LEFT`: rows in this cluster are connected to the right side.
- `RIGHT`: rows in this cluster are connected to the left side.
- `BOTH`: the cluster appears on both sides of a non-symmetric block pair.
- `CLIQUE`: the block is symmetric, so members conflict with the other members in the same block.

This side is important because it tells deletion code which neighbors to update.

## Row-Level Graph

The original implementation is in:

- `initializer.py`
- `graph.py`
- `block_pair.py`

It stores state per row:

- `degrees`: integer array of size `n`
- `active`: boolean array of size `n`
- `deleted`: boolean array of size `n`
- `block_pairs`: symbolic block pairs
- `cluster_to_blocks`: for each cluster, the list of block memberships it participates in
- `row_to_cluster`: row-to-cluster lookup

### Initial Degree Calculation

For each block pair `(L, R)`, and for each cluster `C` involved:

If `C` is only in `L`:

```text
degree[row in C] += number of rows in R
```

If `C` is only in `R`:

```text
degree[row in C] += number of rows in L
```

If `C` is in both sides:

```text
degree[row in C] += number of rows in union(L, R) - 1
```

The `- 1` removes the self-edge. A row should not conflict with itself.

For a symmetric violation, this is the clique case:

```text
degree[row in C] += size_of_clique_block - 1
```

### Vertex Removal

Vanilla repair repeatedly removes the row with max degree.

When row `v` is removed:

1. Find its cluster `C`.
2. Look up `cluster_to_blocks[C]`.
3. For each membership, get the affected neighbor rows:
   - `LEFT` affects right members.
   - `RIGHT` affects left members.
   - `BOTH` or `CLIQUE` affects union members.
4. Filter to currently active rows.
5. Decrement those rows' degrees by one.
6. Mark rows whose degree reaches zero as inactive.
7. Mark `v` as deleted and inactive.

This avoids storing all individual edges, but it can still allocate large temporary arrays like:

```python
active_neighbors = neighbors[self.active[neighbors] & (neighbors != row_idx)]
```

For large duplicate-heavy datasets, those arrays can still be very large.

## Cluster-Count Graph

The newer implementation is in:

- `cluster_count_graph.py`

It uses the same block-pair math, but stores state per cluster rather than per row.

It stores:

- `cluster_sizes`: original number of rows in each cluster
- `remaining`: number of not-yet-deleted rows in each cluster
- `removed`: number of rows already deleted from each cluster
- `degrees`: current degree for any remaining row in the cluster
- `block_pairs`: cluster-level block pairs
- `cluster_to_blocks`: cluster-level memberships

The key observation is:

> All remaining rows in the same compact cluster have the same degree.

They start with the same conflicts, and every deletion affects all rows in that cluster the same way, except when rows from the cluster itself are removed. Therefore we do not need one degree value per row for the greedy max-degree choice. One degree per cluster is enough.

### Initial Degree Calculation

The formulas are the same as the row-level graph, but the graph writes one degree per cluster:

If `C` is only in `L`:

```text
degrees[C] += row_count(R)
```

If `C` is only in `R`:

```text
degrees[C] += row_count(L)
```

If `C` is in both sides:

```text
degrees[C] += row_count(union(L, R)) - 1
```

For a clique:

```text
degrees[C] += row_count(clique_block) - 1
```

### Vertex Removal

The repair loop asks the cluster-count graph for one row at a time:

```python
row_idx = graph.pop_max_degree_vertex()
```

Internally:

1. Choose the active cluster with the largest degree.
2. Emit the next concrete row index from that cluster.
3. Decrease `remaining[cluster]`.
4. For each block membership of that cluster, find affected neighbor clusters.
5. Decrement those neighbor clusters' degrees by one.
6. If a cluster has no remaining rows, set its degree to zero.

No large active-neighbor row arrays are materialized during deletion. The temporary arrays are cluster-id arrays, which are much smaller when the data has many duplicate rows.

## Why This Helps Adult/Tax PATE-CTGAN

The memory issue showed up in vanilla repair for:

- `adult` with `patectgan`
- `tax` with `patectgan`

Those runs can produce large datasets with many repeated compact rows and very large conflict blocks. The old graph avoids explicit edges, but deletion can still repeatedly allocate huge row-neighbor arrays.

The cluster-count graph keeps the same conflict math but runs the greedy cover loop at cluster-count granularity. This should reduce memory pressure most when:

- many rows collapse into fewer compact clusters,
- conflict blocks are large,
- repeated deletion would otherwise allocate large active-neighbor arrays.

## When Each Graph Is Used

By default, vanilla repair uses the cluster-count graph only for:

```python
dataset_name in {"adult", "tax"} and generation_model == "patectgan"
```

Everything else still uses the original row-level graph.

This gate is in:

```python
vanilla_vc_repair/vanilla_vc_repair.py
```

The environment variable `VANILLA_REPAIR_CLUSTER_COUNT_GRAPH` can override the gate:

```bash
VANILLA_REPAIR_CLUSTER_COUNT_GRAPH=legacy
```

forces the original row-level graph.

```bash
VANILLA_REPAIR_CLUSTER_COUNT_GRAPH=all
```

forces the cluster-count graph for all datasets.

## Correctness Notes

The two implementations represent the same block-pair conflict graph. The important invariants are:

1. A cluster's degree equals the degree of any remaining row in that cluster.
2. Removing one row from cluster `C` reduces each active neighbor row's degree by exactly one.
3. In the cluster-count graph, this same update is represented by reducing each affected neighbor cluster's degree by one.
4. Overlapping blocks use `union(L, R)` so a cluster that appears on both sides is not counted twice.
5. Clique blocks subtract one to exclude self-conflicts.

There can be tie differences. If two rows or clusters have the same max degree, the old and new graph may choose different concrete row ids. The expected invariant is that the greedy cover size and edge-removal behavior match for equivalent compact conflict structure.

The test file `tests/test_graph.py` covers:

- biclique degree initialization and deletion updates,
- symmetric clique behavior,
- overlapping block pairs,
- random-edge sampling for the row-level graph,
- cluster-count cover behavior for biclique, clique, and overlap cases.

