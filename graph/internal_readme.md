# RPM Conflict Graph: Biclique-Compressed Engine

The `Graph` is a high-scale engine for tracking dataset conflicts. It implements the symbolic mental model from `how_i_picture_graph.md` using **Biclique Compression** to achieve extreme performance by exploiting the redundancy in multi-row violations.

---

## 1. The Core Optimization: Biclique Compression

In this graph, vertices are grouped into **Clusters**, and conflicts are defined as **Blocks** (Bicliques or Cliques between clusters). Instead of tracking millions of individual edges, we track symbolic relationships between groups of rows.

### Key Performance Win
When a vertex is removed, we don't iterate over its edges. Instead:
1. We lookup which **Blocks** the vertex belongs to using a cluster-to-block "Phonebook".
2. We perform a **Batch Update** on all vertices on the "other side" of those blocks using NumPy vectorization.

This reduces the complexity of degree updates from $O(\text{edges})$ to $O(\text{connected blocks} \times 1)$, where "1" is a single vectorized CPU instruction.

---

## 2. Technical Implementation

### Module Split
- `block_pair.py`: Defines the symbolic conflict blocks and side-specific memberships (`LEFT`, `RIGHT`, `BOTH`, `CLIQUE`).
- `cluster_map.py`: Validates and stores row-to-cluster and cluster-to-row mappings.
- `types.py`: Defines the small violation protocols consumed by the graph package.
- `initializer.py`: Calculates initial vertex degrees and builds the cluster-to-block index.
- `graph.py`: The runtime engine. Manages vertex degrees and the active/deleted masks.
- `builder.py`: Public entry point for constructing a `Graph` from any compatible violation set.

### A. Initialization
1.  **Map Clusters**: Vertices are assigned to clusters via `ClusterMap`.
2.  **Initial Degrees**: For each block, we iterate over the clusters involved. Depending on whether a cluster is on the `LEFT`, `RIGHT`, or `BOTH` sides, we increment the degrees of its members by the size of the "other side".
3.  **Cluster Index**: Each cluster stores a list of `BlockMembership` objects pointing to the blocks it participates in and its role (`side`).

### B. Efficient Vertex Removal
When `vertex_i` (in cluster $C$) is removed:
1.  **Direct Lookup**: Jump to the specific blocks connected to cluster $C$ via the cluster-to-block index.
2.  **Side-Aware Neighbors**: The `BlockMembership` provides the correct "other side" members (e.g., if on `LEFT`, neighbors are on `RIGHT`; if on `BOTH`, neighbors are the `union` of both sides).
3.  **Vectorized Update**: Decrement the degrees of all **active** neighbors in one shot: `degrees[active_neighbors] -= 1`.
4.  **State Update**: Update the `active` mask for any vertices whose degree reached zero.

---

## 3. Handling Overlaps (`BOTH`)
If a cluster appears on both sides of a biclique (an overlapping biclique), it is marked as `BOTH`. 
- **Initialization**: Its members' degrees are increased by `|union(L, R)| - 1`.
- **Removal**: Its neighbors are defined as `union(L, R)`, and the `active` filter ensures it doesn't decrement its own degree twice.

---

## 4. Performance Features
- **Zero Redundant Concatenation**: Block member arrays and unions are pre-calculated during initialization.
- **$O(1)$ Search**: The Phonebook eliminates the need to scan blocks during the repair loop.
- **NumPy Native**: All heavy lifting is done in C-optimized NumPy operations.
