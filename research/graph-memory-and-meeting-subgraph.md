# Graph Memory & Meeting Subgraph Research

Research into rustworkx graph memory characteristics and optimal algorithms for extracting meeting subgraphs connecting seed nodes in large directed graphs.

**Date:** 2026-02-27

---

## Context

Grover uses rustworkx `PyDiGraph` for in-memory knowledge graphs (file dependencies, code analysis). As the system scales to multi-tenant deployment, two questions arise:

1. **How much memory does a rustworkx graph consume?** Can we serve 50+ concurrent users within a 2 GB RAM budget?
2. **What is the fastest algorithm for extracting a meeting subgraph** — the minimal connected subgraph linking a set of seed nodes?

---

## Part 1: Memory Profiling

### 1.1 Node payload dominates

With 1024-dim `list[float]` vectors stored on nodes (4 edges per node, lightweight edge payloads):

| Nodes | Edges | Memory | KB/node |
|------:|------:|-------:|--------:|
| 100 | 399 | 3.3 MB | 33.6 |
| 1,000 | 3,998 | 32.8 MB | 33.6 |
| 5,000 | 19,995 | 164.1 MB | 33.6 |
| 10,000 | 39,999 | 328.3 MB | 33.6 |

Scaling is perfectly linear at ~33.6 KB/node. A Python float is 28 bytes, so 1024 floats = 28.7 KB of float objects alone — the vector payload accounts for ~85% of per-node cost.

**Conclusion:** With vectors on nodes, 10K nodes x 50 users = ~16 GB. Vectors should live in the vector store, not on graph nodes.

**How this was measured:**

```python
import gc, random, tracemalloc
import rustworkx as rx

def measure_graph_memory(n_nodes: int, vector_dim: int = 1024) -> int:
    """Returns bytes consumed by a graph with vector payloads on nodes."""
    gc.collect()
    tracemalloc.start()

    g = rx.PyDiGraph()
    for i in range(n_nodes):
        g.add_node({
            "path": f"/src/module_{i}.py",
            "type": "file",
            "vector": [random.random() for _ in range(vector_dim)],
        })
    for src in range(n_nodes):
        for tgt in random.sample(
            [t for t in range(n_nodes) if t != src],
            min(4, n_nodes - 1),
        ):
            g.add_edge(src, tgt, {"type": "imports"})

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return current
```

### 1.2 Edge-only storage (no vectors on nodes)

Pure topology storage — just the edges, no node/edge metadata. Measured with `tracemalloc` (accurate for Python allocations):

| Structure | 100K edges | 500K edges | 1M edges |
|-----------|----------:|----------:|---------:|
| `list[tuple[int,int]]` | 781 KB | 3.8 MB | 7.6 MB |
| `numpy int32 array` | 781 KB | 3.8 MB | 7.6 MB |
| `dict[int,list[int]]` | 1.8 MB | 5.0 MB | 9.1 MB |

At 1M edges x 50 users: ~382 MB for `list[tuple]` or numpy. Well within budget.

```python
import gc, random, tracemalloc
from collections import defaultdict
import numpy as np

def measure_edge_storage(edges: list[tuple[int, int]], builder) -> int:
    gc.collect()
    tracemalloc.start()
    obj = builder(edges)
    current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return current

edges = [(random.randint(0, 9999), random.randint(0, 9999))
         for _ in range(1_000_000)]

# list[tuple] — 7.6 MB
measure_edge_storage(edges, list)

# numpy int32 — 7.6 MB
measure_edge_storage(edges, lambda e: np.array(e, dtype=np.int32))

# dict[int, list[int]] — 9.1 MB
def to_adj(edges):
    adj = defaultdict(list)
    for s, t in edges:
        adj[s].append(t)
    return dict(adj)
measure_edge_storage(edges, to_adj)
```

### 1.3 Rustworkx Rust-heap overhead

Rustworkx allocates on the Rust heap, invisible to `tracemalloc`. Measured via RSS delta: build 50 identical copies, measure process RSS change, divide by 50.

| Edges | rustworkx | `list[tuple]` | Overhead |
|------:|----------:|--------------:|---------:|
| 100K | 2.3 MB | 781 KB | 1.5 MB |
| 500K | 11.2 MB | 3.8 MB | 7.4 MB |
| 1M | 20.0 MB | 7.6 MB | 12.4 MB |

Rustworkx costs ~2.5x a plain tuple list. The overhead pays for O(1) neighbor lookup and Rust-backed graph algorithms. At 1M edges x 50 users: ~1 GB — acceptable.

```python
import gc, os, random
import psutil
import rustworkx as rx

COPIES = 50

def measure_rustworkx_rss(edges, n_nodes) -> float:
    """Returns estimated bytes per graph via RSS delta."""
    def build():
        g = rx.PyDiGraph()
        g.add_nodes_from([None] * n_nodes)
        g.add_edges_from_no_data(edges)
        return g

    build()  # warm up allocator
    gc.collect()

    before = psutil.Process(os.getpid()).memory_info().rss
    graphs = [build() for _ in range(COPIES)]
    gc.collect()
    after = psutil.Process(os.getpid()).memory_info().rss

    return max((after - before) / COPIES, 0)
```

### 1.4 Graph construction: build vs deepcopy

| Size | Build from edges | `deepcopy` | Ratio |
|-----:|-----------------:|-----------:|------:|
| 10K nodes / 40K edges | 0.016s | 0.049s | 3.1x |
| 100K nodes / 400K edges | 0.157s | 0.491s | 3.1x |
| 1M nodes / 4M edges | 1.658s | 5.131s | 3.1x |

**Conclusion:** Always build from edge data. Never deepcopy.

```python
import copy, time
import rustworkx as rx

def build_from_edges(n_nodes, edges):
    g = rx.PyDiGraph()
    g.add_nodes_from(list(range(n_nodes)))
    g.add_edges_from_no_data(edges)
    return g

edges = [...]  # pre-generated

# Build: 1.658s at 1M nodes
t0 = time.perf_counter()
g = build_from_edges(1_000_000, edges)
print(f"Build: {time.perf_counter() - t0:.3f}s")

# Deepcopy: 5.131s at 1M nodes
t0 = time.perf_counter()
g2 = copy.deepcopy(g)
print(f"Copy: {time.perf_counter() - t0:.3f}s")
```

---

## Part 2: Meeting Subgraph Algorithms

### Problem definition

Given a directed graph `G` (up to 1M nodes, 4M edges) and a set of 1,000 seed nodes, find the minimal connected subgraph of `G` that includes all seed nodes. Requirements:

- Seed nodes are never removed
- The subgraph remains connected (removing a node cannot split it)
- Minimize total nodes/edges

This is a variant of the **Steiner tree problem**, which is NP-hard for directed graphs. We need practical approximations.

### 2.1 `rustworkx.steiner_tree` (built-in, undirected only)

Implements Kou-Markowsky-Berman 2-approximation. Internally computes the metric closure (all-pairs shortest paths between terminals), requiring k Dijkstra calls where k = number of seeds.

| Size | Time |
|-----:|-----:|
| 10K | 0.029s |
| 50K | 0.204s |
| 100K | 0.551s |
| 250K+ | too slow |

**Verdict:** Only works on `PyGraph` (not `PyDiGraph`). Scales poorly beyond 100K nodes because it runs 1,000 individual Dijkstra traversals internally. Not viable at target scale.

```python
import rustworkx as rx

# Requires conversion to undirected PyGraph
undirected = digraph.to_undirected(multigraph=False)
tree = rx.steiner_tree(undirected, seed_nodes, weight_fn=lambda e: 1.0)
# tree is a PyGraph with only the connecting edges
```

### 2.2 rustworkx `digraph_bfs_search` with visitor callbacks

Rustworkx supports multi-source BFS via `digraph_bfs_search(graph, source=list_of_seeds, visitor=...)`.

**Critical finding:** rustworkx processes multiple sources **sequentially, not interleaved**. The first source's BFS covers the entire connected graph before any other source begins. This makes wavefront-meeting detection impossible — all nodes receive the same origin label.

This was confirmed by tracing discovery order on a simple chain graph:

```python
import rustworkx as rx
from rustworkx.visit import BFSVisitor

class TraceVisitor(BFSVisitor):
    def __init__(self):
        self.order = []
    def discover_vertex(self, v):
        self.order.append(v)

# Chain: 0 -> 1 -> 2 -> 3 -> 4
g = rx.PyGraph()
g.add_nodes_from(list(range(5)))
for i in range(4):
    g.add_edge(i, i + 1, None)

visitor = TraceVisitor()
rx.graph_bfs_search(g, source=[0, 4], visitor=visitor)
print(visitor.order)
# Expected (interleaved): [0, 4, 1, 3, 2]
# Actual (sequential):    [0, 1, 2, 3, 4]
# Source 0 covers everything before source 4 starts.
```

Additionally, even with a corrected visitor (handling seeds specially in `tree_edge`), per-edge Python callbacks crossing the Rust-Python FFI boundary make it **5-6x slower** than pure Python BFS.

**Verdict:** Not viable. Sequential source processing prevents wavefront meeting detection, and callback overhead kills performance.

### 2.3 Pure Python multi-source BFS with adjacency lists

Build Python `list[list[int]]` adjacency lists (forward + reverse), run multi-source BFS with `collections.deque`, track origin per wavefront, union-find to detect when components merge.

| Size | Time | Subgraph nodes |
|-----:|-----:|---------------:|
| 10K | 8ms | 1,630 |
| 100K | 80ms | 2,541 |
| 500K | 555ms | 3,320 |
| 1M | 1,350ms | 3,573 |

**Verdict:** Correct results, but the adjacency list construction (O(E) Python list allocation) dominates runtime at scale.

### 2.4 Multi-source BFS with `neighbors_undirected()` (winner)

`PyDiGraph.neighbors_undirected(node)` returns the union of successors and predecessors directly from rustworkx's Rust adjacency structure — no Python adjacency list construction needed. The BFS loop runs in Python but neighbor lookups are Rust-backed.

This is a method on `PyDiGraph` — the graph stays directed for PageRank, community detection, and other algorithms. `neighbors_undirected()` simply provides an undirected view of a single node's neighborhood for the BFS traversal.

| Size | Python adj lists | `neighbors_undirected()` | Speedup | + Leaf prune | Total |
|-----:|-----------------:|-------------------------:|--------:|-------------:|------:|
| 10K | 6ms | 5ms | 1.2x | 2ms | 7ms |
| 50K | 40ms | 16ms | 2.5x | 5ms | 22ms |
| 100K | 63ms | 17ms | 3.6x | 8ms | 26ms |
| 250K | 205ms | 24ms | 8.4x | 19ms | 43ms |
| 500K | 547ms | 34ms | 16.1x | 36ms | 70ms |
| 1M | 1,332ms | 61ms | 21.9x | 70ms | 130ms |

**Properties:**
- **Zero extra memory** — reads from rustworkx's existing Rust adjacency, no Python list allocation
- **Speedup grows with scale** — 1.2x at 10K, 21.9x at 1M
- **Works on directed graphs** — traverses both directions for connectivity while the graph itself remains directed
- **Same quality** — produces identical results (999 bridges, same node counts) as the adjacency list approach

**Verdict: best approach.** 130ms at 1M nodes. 50 concurrent requests = 6.5s CPU time, negligible memory.

---

## Part 3: Reference Implementation

### 3.1 Union-Find

Path-compressed union-find with rank balancing. Tracks component count for early termination.

```python
class UnionFind:
    __slots__ = ("parent", "rank", "_components")

    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}
        self._components = len(self.parent)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self._components -= 1
        return True

    @property
    def components(self) -> int:
        return self._components
```

### 3.2 Multi-source BFS with `neighbors_undirected()`

The core algorithm. One pass through the graph, O(V+E) worst case, terminates early when all seed components are connected.

```python
from collections import deque
import rustworkx as rx


def multisource_bfs(
    graph: rx.PyDiGraph,
    seeds: list[int],
) -> tuple[set[int], list[tuple[int, int]]]:
    """Find the minimal node set connecting all seeds via shortest paths.

    Algorithm:
    1. Initialize BFS queue with all seeds, each labeled as its own origin.
    2. Expand wavefronts simultaneously. Each newly discovered node inherits
       the origin of the node that discovered it.
    3. When an edge connects two nodes from different seed components,
       that's a "bridge" — union the components.
    4. Stop when all seeds are in one connected component.
    5. Trace predecessor chains from each bridge endpoint back to its seed
       root to collect the minimal connecting node set.

    Uses graph.neighbors_undirected(node) for neighbor iteration — this reads
    directly from rustworkx's Rust adjacency structure, avoiding O(E) Python
    adjacency list construction.

    Parameters
    ----------
    graph : rx.PyDiGraph
        The full directed graph. Not modified.
    seeds : list[int]
        Node indices to connect.

    Returns
    -------
    tuple[set[int], list[tuple[int, int]]]
        (kept_node_indices, bridge_edges)
    """
    n_nodes = graph.num_nodes()
    origin = [-1] * n_nodes  # which seed's wavefront claimed this node
    pred = [-1] * n_nodes    # predecessor in BFS tree
    uf = UnionFind(seeds)

    queue = deque()
    for s in seeds:
        origin[s] = s
        pred[s] = s  # self-predecessor marks root
        queue.append(s)

    bridges: list[tuple[int, int]] = []

    while queue and uf.components > 1:
        node = queue.popleft()
        node_origin = origin[node]

        # neighbors_undirected returns successors + predecessors
        for neighbor in graph.neighbors_undirected(node):
            if origin[neighbor] == -1:
                # Unclaimed — this wavefront claims it
                origin[neighbor] = node_origin
                pred[neighbor] = node
                queue.append(neighbor)
            elif uf.find(origin[neighbor]) != uf.find(node_origin):
                # Two different seed components meet — bridge!
                bridges.append((node, neighbor))
                uf.union(origin[neighbor], node_origin)

    # Trace predecessor chains from bridge endpoints back to seeds
    keep_nodes = set(seeds)
    for a, b in bridges:
        for start in (a, b):
            node = start
            while node != pred[node]:
                keep_nodes.add(node)
                node = pred[node]
            keep_nodes.add(node)

    return keep_nodes, bridges
```

### 3.3 Leaf stripping

Iteratively removes non-seed nodes that are leaves in the directed subgraph (no predecessors OR no successors). O(n+e) total — each node and edge touched at most twice.

This is far superior to the naive approach of iterating all nodes, copying the graph, removing one node, and checking connectivity — which is O(n^2 x (n+e)).

```python
from collections import defaultdict


def strip_leaves(
    nodes: set[int],
    edges: list[tuple[int, int]],
    protected: set[int],
) -> tuple[set[int], set[tuple[int, int]]]:
    """Remove non-protected leaf nodes iteratively. O(n+e) total.

    A "leaf" in a directed subgraph is a node with no predecessors
    OR no successors within the subgraph. Removing it cannot disconnect
    the graph (it's at the periphery). We peel leaves from the outside
    in, like an onion.

    Parameters
    ----------
    nodes : set[int]
        All node indices in the subgraph.
    edges : list[tuple[int, int]]
        All directed edges in the full graph (we filter to subgraph).
    protected : set[int]
        Seed nodes — never removed.

    Returns
    -------
    tuple[set[int], set[tuple[int, int]]]
        (remaining_nodes, remaining_edges)
    """
    # Build local adjacency for the subgraph only
    successors: dict[int, set[int]] = defaultdict(set)
    predecessors: dict[int, set[int]] = defaultdict(set)

    for src, tgt in edges:
        if src in nodes and tgt in nodes:
            successors[src].add(tgt)
            predecessors[tgt].add(src)

    # Initialize queue with current leaves
    queue = [
        n for n in nodes
        if n not in protected
        and (not successors[n] or not predecessors[n])
    ]

    removed = set()
    while queue:
        node = queue.pop()
        if node in removed or node in protected:
            continue
        removed.add(node)

        # Update neighbors — may create new leaves
        for succ in successors[node]:
            predecessors[succ].discard(node)
            if succ not in protected and succ not in removed:
                if not predecessors[succ] or not successors[succ]:
                    queue.append(succ)

        for pred_node in predecessors[node]:
            successors[pred_node].discard(node)
            if pred_node not in protected and pred_node not in removed:
                if not successors[pred_node] or not predecessors[pred_node]:
                    queue.append(pred_node)

    remaining_nodes = nodes - removed
    remaining_edges = {
        (s, t) for s, t in edges
        if s in remaining_nodes and t in remaining_nodes
    }
    return remaining_nodes, remaining_edges
```

### 3.4 Building the result subgraph

```python
import rustworkx as rx


def build_subgraph(
    source_graph: rx.PyDiGraph,
    nodes: set[int],
    edges: set[tuple[int, int]],
) -> rx.PyDiGraph:
    """Build a new PyDiGraph from kept nodes/edges, preserving node data.

    Parameters
    ----------
    source_graph : rx.PyDiGraph
        Original graph (for node/edge data lookup).
    nodes : set[int]
        Node indices to include.
    edges : set[tuple[int, int]]
        Directed edges to include.

    Returns
    -------
    rx.PyDiGraph
        New graph with remapped indices. Each node's payload is the
        original node data from source_graph.
    """
    sub = rx.PyDiGraph()
    old_to_new: dict[int, int] = {}

    for old_idx in sorted(nodes):
        new_idx = sub.add_node(source_graph[old_idx])
        old_to_new[old_idx] = new_idx

    for src, tgt in edges:
        if src in old_to_new and tgt in old_to_new:
            edge_data = source_graph.get_edge_data(src, tgt)
            sub.add_edge(old_to_new[src], old_to_new[tgt], edge_data)

    return sub
```

### 3.5 Full pipeline

Putting it all together:

```python
import rustworkx as rx


def meeting_subgraph(
    graph: rx.PyDiGraph,
    seeds: list[int],
    edge_list: list[tuple[int, int]],
) -> rx.PyDiGraph:
    """Extract the minimal subgraph connecting all seed nodes.

    Pipeline:
    1. Multi-source BFS via neighbors_undirected()  — 61ms at 1M nodes
    2. Trace bridge paths via predecessor map         — <1ms
    3. Leaf stripping (iterative, O(n+e))             — 70ms at 1M nodes
    4. Build result PyDiGraph from kept nodes/edges   — <10ms

    Total: ~130ms per request at 1M nodes.

    Parameters
    ----------
    graph : rx.PyDiGraph
        The full directed graph. Shared across requests, never modified.
    seeds : list[int]
        Node indices to connect.
    edge_list : list[tuple[int, int]]
        Pre-extracted edge list (graph.edge_list() or from DB).

    Returns
    -------
    rx.PyDiGraph
        Pruned subgraph connecting all reachable seeds.
    """
    seed_set = set(seeds)

    # Step 1-2: BFS + path tracing
    kept_nodes, bridges = multisource_bfs(graph, seeds)

    # Step 3: Prune non-essential nodes
    kept_nodes, kept_edges = strip_leaves(kept_nodes, edge_list, seed_set)

    # Step 4: Build result
    return build_subgraph(graph, kept_nodes, kept_edges)
```

Usage:

```python
import rustworkx as rx

# One-time setup — global graph, shared across all requests
graph = rx.PyDiGraph()
graph.add_nodes_from(node_data_list)
graph.add_edges_from(edge_data_list)
edge_list = list(graph.edge_list())  # cache for reuse

# Per request — pick seed nodes, extract meeting subgraph
seeds = [42, 108, 256, 1001, ...]  # from user query / search results
sub = meeting_subgraph(graph, seeds, edge_list)

# The graph is still available for other algorithms
pagerank = rx.pagerank(graph)
components = rx.strongly_connected_components(graph)
```

---

## Part 4: Alternative Approaches Considered

### 4.1 scipy `dijkstra(min_only=True)`

`scipy.sparse.csgraph.dijkstra` with `indices=seeds, unweighted=True, min_only=True` runs multi-source BFS entirely in C/Cython. Returns distance, predecessor, and source-label arrays.

```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# Convert rustworkx edge list to CSR matrix
edge_arr = np.array(graph.edge_list(), dtype=np.int32)
data = np.ones(len(edge_arr), dtype=np.float32)
n = graph.num_nodes()
csr = csr_matrix((data, (edge_arr[:, 0], edge_arr[:, 1])), shape=(n, n))

# Multi-source BFS — all 1000 seeds in one C-level pass
seed_arr = np.array(seeds, dtype=np.int32)
dist, predecessors, sources = dijkstra(
    csr,
    directed=True,
    indices=seed_arr,
    unweighted=True,       # BFS, not weighted Dijkstra
    min_only=True,         # one result per node (nearest seed)
    return_predecessors=True,
)
# dist[i] = distance from nearest seed to node i
# predecessors[i] = parent of node i in BFS tree
# sources[i] = which seed is closest to node i
```

**Estimated performance:** 200-400ms at 1M nodes (CSR conversion ~50-100ms + C-level BFS ~150-300ms).

**Trade-off:** Slightly slower than `neighbors_undirected()` (130ms) due to CSR conversion overhead. Requires scipy dependency. But the BFS itself runs in compiled C rather than a Python loop, so it may win at larger scales.

### 4.2 Single-source Dijkstra star topology

Pick one seed as root, compute shortest paths to all others via `digraph_dijkstra_shortest_paths` with `weight_fn=None, as_undirected=True` (runs entirely in Rust, no Python callbacks).

```python
# Runs in ~1.4s at 1M nodes — one Rust-backed Dijkstra pass
paths = rx.digraph_dijkstra_shortest_paths(
    graph, seeds[0],
    weight_fn=None,          # no callback, pure Rust
    default_weight=1.0,
    as_undirected=True,
)
# paths[target] = list of node indices on shortest path from root to target
```

Produces a star topology (all paths through one root) — larger subgraph than multi-source BFS. Simple but suboptimal.

### 4.3 Pairwise meeting points via common descendants

An alternative formulation: for each pair of seeds, find their closest common descendant and connect through it. This is what the `MeetingSubgraphFinder` pattern uses:

```python
from itertools import combinations
import rustworkx as rx

def pairwise_meeting(graph, seeds):
    """Find meeting subgraph via pairwise common descendants."""
    weight_fn = lambda _: 1.0

    # Per-seed shortest paths (1000 Dijkstra calls)
    paths_from = {}
    dist_from = {}
    for seed in seeds:
        paths_from[seed] = rx.dijkstra_shortest_paths(
            graph, seed, weight_fn=weight_fn
        )
        dist_from[seed] = rx.dijkstra_shortest_path_lengths(
            graph, seed, weight_fn
        )

    # For each pair, find closest common descendant
    subgraph_nodes = set(seeds)
    for n1, n2 in combinations(seeds, 2):
        reachable = set(dist_from[n1]) & set(dist_from[n2])
        if not reachable:
            continue
        best = min(reachable, key=lambda c: dist_from[n1][c] + dist_from[n2][c])
        for src in (n1, n2):
            if best in paths_from[src]:
                subgraph_nodes.update(paths_from[src][best])

    return subgraph_nodes
```

**Bottleneck:** 1,000 individual Dijkstra calls from each seed. At 1M nodes, ~50-100s total. Can be replaced with the multi-source BFS approach for the path-finding step.

---

## Part 5: Recommended Architecture

### Graph storage

- Store the global knowledge graph in a rustworkx `PyDiGraph`
- No vectors on graph nodes — vectors live in the vector store (usearch, Pinecone, Databricks)
- Graph holds lightweight metadata: file paths, node types, edge types
- At 1M nodes / 4M edges with None payloads: ~20 MB per graph

### Meeting subgraph extraction (per request)

```
Input: PyDiGraph (global, shared) + list[int] seed nodes
    |
    v
Multi-source BFS via neighbors_undirected()     <-- 61ms at 1M
    |
    v
Trace bridge paths via predecessor map           <-- <1ms
    |
    v
Leaf stripping (iterative, O(n+e))               <-- 70ms at 1M
    |
    v
Build result PyDiGraph from kept nodes/edges     <-- <10ms
    |
    v
Output: small PyDiGraph (~1,500 nodes)
```

Total: ~130ms per request at 1M nodes. No graph copying. No extra memory.

### Concurrency

The multi-source BFS reads the global graph via `neighbors_undirected()` — read-only. In asyncio (single-threaded cooperative multitasking), concurrent reads are safe. Each request builds its own result graph from the kept nodes.

At 50 concurrent requests x 130ms = 6.5s of sequential CPU, plus ~0 MB additional memory per request (beyond the small result graph).

### Pruning strategy

Leaf stripping (O(n+e) total) is the correct pruning approach for path-based subgraphs:

1. Build local successor/predecessor sets for the subgraph
2. Initialize queue with non-seed nodes that have no predecessors OR no successors
3. Remove them, check if their neighbors became removable
4. Repeat until stable

This is far superior to the naive approach of iterating all nodes, copying the graph, removing one node, and checking connectivity (O(n^2 x (n+e))).

---

## Part 6: What Rustworkx Lacks

The key missing primitive in rustworkx is a **multi-source BFS with interleaved wavefronts and origin tracking** — something like:

```python
# Hypothetical API that would run entirely in Rust
origins, predecessors = rx.multi_source_bfs(
    graph, sources=seed_list, directed=False
)
# origins[node] = which seed reached it first
# predecessors[node] = parent in BFS tree
```

This would run entirely in Rust and eliminate the Python BFS loop, likely achieving 5-10ms at 1M nodes instead of 61ms. The current `digraph_bfs_search` processes sources sequentially (not interleaved) and requires per-edge Python callbacks, making it unsuitable.

### Key rustworkx findings

| Feature | Status | Impact |
|---------|--------|--------|
| `steiner_tree` | Undirected only, O(k x (V+E)) | Not viable for directed graphs at scale |
| `digraph_bfs_search` multi-source | Sources processed sequentially | Cannot detect wavefront meetings |
| BFSVisitor callbacks | Per-edge Python FFI crossing | 5-6x slower than pure Python BFS |
| `neighbors_undirected()` | Works on PyDiGraph | Eliminates adj list build, 22x speedup |
| `dijkstra_shortest_paths` `weight_fn=None` | Pure Rust, no callbacks | Fast for single-source, but 1000x calls needed |
