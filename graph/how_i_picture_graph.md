## How I Picture the Graph
There are n vertices. 
Vertices are combined to clusters
Each cluster has an index and a list of members (the vertices)
We say that two clusters $C_1, C_2$ are connected if every pair $v_1,v_2\in C_1\times C_2$ is an edge in the graph.
A block is a list of clusters. 
We say that two blocks $B_1, B_2$ are connected if every cluster pair $c_1,c_2\in B_1\times B_2$ is connected. 
The graph is built using n and a set of block pairs. 
The graph holds the following: 
- degrees: Int array of size n representing the degree of each vertex at any time
- active: Bool array of size n representing which vertex is active at any time 
- deleted: Bool array of size n representing which vertex were deleted from the data 
- cluster_mapping: Object containing vertex_to_cluster and cluster_to_vertices functions to allow simple conversion 
- blocks_pairs: List of block pairs 
- cluster_to_blocks: A dictionary with keys corresponding to the cluster indices and values are list of tuples block_pairs indices that the cluster is a part of and the "side" (can be left, right or both).

Init (input: int n, a set of block pairs B, and a cluster mapping): 
1. Set deg = [0,...,0]
2. Set cluster_to_block = defaultdict([])
3. For v in range(0, n)
    1. Let c be the cluster of v (using cluster mapping)
    2. For i, (L,R) in enumerate(B):
        1. Set in_left = c in L 
        2. Set in_right = c in R
        3. If in_left and in_right: 
            1. deg[v] += union(L, R).count_members() - 1
            2. clusters_to_blocks[c].append((i,BOTH))
        4. Else if in_left: 
            1. deg[v] += R.count_members()
            2. clusters_to_blocks[c].append((i,LEFT))
        5. Else if c in R: 
            1. deg[v] += L.count_members()
            2. clusters_to_blocks[c].append((i,RIGHT))
4. Set active = deg > 0 
5. Set deleted = [False,..., False]
6. Save B as blocks_pairs and the cluster_mapping. 

The needed API for the graph is: 
* Any edges
* Degree of each vertex
* Remove a vertex

Any edges could be simply calculated by checking if there is an active vertex.
Degree of vertex could be simply given by just returning the value of deg[v]
Removing a vertex would be calculated as follows: 
Delete vertex(v):
1. If deleted[v] or not active[v] then return
2. Set c to be the cluster of v (using the cluster_mapping)
3. For i, side in cluster_to_block: 
    1. vertices = L if side == RIGHT, R if side == LEFT, union(L,R) if side == BOTH
    2. vertices = filter vertices according to active
    3. deg[vertices] -= 1
    4. active[vertices] = deg[vertices] > 0
5. deleted[v] = True
6. active[v] = False
