"""RustworkxGraph — rustworkx-backed graph store implementing GraphStore protocol."""

from __future__ import annotations

import json
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any

import rustworkx

from grover.graph.types import SubgraphResult, subgraph_result
from grover.ref import Ref

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class RustworkxGraph:
    """Directed knowledge graph over file paths.

    Wraps a ``rustworkx.PyDiGraph`` with string-path-keyed nodes and provides
    traversal queries (dependents, impacts, path_between) plus async
    persistence to/from the ``grover_edges`` / ``grover_files`` tables.

    Implements the ``GraphStore`` and ``SupportsPersistence`` protocols.
    """

    def __init__(self) -> None:
        self._graph: rustworkx.PyDiGraph = rustworkx.PyDiGraph()
        self._path_to_idx: dict[str, int] = {}
        self._idx_to_path: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, path: str, **attrs: Any) -> None:
        """Add or update a node.  Merges *attrs* if the node already exists."""
        if path in self._path_to_idx:
            idx = self._path_to_idx[path]
            existing: dict[str, Any] = self._graph[idx]
            existing.update(attrs)
        else:
            data: dict[str, Any] = {"path": path, **attrs}
            idx = self._graph.add_node(data)
            self._path_to_idx[path] = idx
            self._idx_to_path[idx] = path

    def remove_node(self, path: str) -> None:
        """Remove a node and all incident edges.  Raises ``KeyError`` if missing."""
        idx = self._require_node(path)
        self._graph.remove_node(idx)
        del self._path_to_idx[path]
        del self._idx_to_path[idx]

    def has_node(self, path: str) -> bool:
        """Return whether *path* is in the graph."""
        return path in self._path_to_idx

    def get_node(self, path: str) -> dict[str, Any]:
        """Return the node data dict.  Raises ``KeyError`` if missing."""
        idx = self._require_node(path)
        return dict(self._graph[idx])

    def nodes(self) -> list[str]:
        """Return all node paths."""
        return list(self._path_to_idx.keys())

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        *,
        weight: float = 1.0,
        edge_id: str | None = None,
        **attrs: Any,
    ) -> None:
        """Add or upsert a directed edge.

        Auto-creates missing endpoint nodes.  On upsert the original ``id`` is
        preserved and *attrs* are merged into ``metadata``.
        """
        # Auto-create nodes
        if not self.has_node(source):
            self.add_node(source)
        if not self.has_node(target):
            self.add_node(target)

        src_idx = self._path_to_idx[source]
        tgt_idx = self._path_to_idx[target]

        # Check for existing edge
        existing_idx = self._find_edge_idx(src_idx, tgt_idx)
        if existing_idx is not None:
            data: dict[str, Any] = self._graph.get_edge_data_by_index(existing_idx)
            data["type"] = edge_type
            data["weight"] = weight
            data["metadata"].update(attrs)
        else:
            resolved_id = edge_id or str(uuid.uuid4())
            data = {
                "id": resolved_id,
                "source": source,
                "target": target,
                "type": edge_type,
                "weight": weight,
                "metadata": dict(attrs),
            }
            self._graph.add_edge(src_idx, tgt_idx, data)

    def remove_edge(self, source: str, target: str) -> None:
        """Remove the edge between *source* and *target*.  Raises ``KeyError``."""
        src_idx = self._require_node(source)
        tgt_idx = self._require_node(target)
        edge_idx = self._find_edge_idx(src_idx, tgt_idx)
        if edge_idx is None:
            msg = f"No edge from {source!r} to {target!r}"
            raise KeyError(msg)
        self._graph.remove_edge_from_index(edge_idx)

    def has_edge(self, source: str, target: str) -> bool:
        """Return ``True`` if the edge exists.  ``False`` if nodes are missing."""
        src_idx = self._path_to_idx.get(source)
        tgt_idx = self._path_to_idx.get(target)
        if src_idx is None or tgt_idx is None:
            return False
        return self._find_edge_idx(src_idx, tgt_idx) is not None

    def get_edge(self, source: str, target: str) -> dict[str, Any]:
        """Return edge data dict.  Raises ``KeyError`` if missing."""
        src_idx = self._require_node(source)
        tgt_idx = self._require_node(target)
        edge_idx = self._find_edge_idx(src_idx, tgt_idx)
        if edge_idx is None:
            msg = f"No edge from {source!r} to {target!r}"
            raise KeyError(msg)
        return dict(self._graph.get_edge_data_by_index(edge_idx))

    def edges(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Return all edges as ``(source, target, data)`` triples."""
        result: list[tuple[str, str, dict[str, Any]]] = []
        for src_idx, tgt_idx, data in self._graph.weighted_edge_list():
            src = self._idx_to_path.get(src_idx, "")
            tgt = self._idx_to_path.get(tgt_idx, "")
            result.append((src, tgt, dict(data)))
        return result

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def dependents(self, path: str) -> list[Ref]:
        """Nodes with edges pointing *to* this node (predecessors)."""
        idx = self._require_node(path)
        return [Ref(path=self._idx_to_path[p]) for p in self._graph.predecessor_indices(idx)]

    def dependencies(self, path: str) -> list[Ref]:
        """Nodes this node points *to* (successors)."""
        idx = self._require_node(path)
        return [Ref(path=self._idx_to_path[s]) for s in self._graph.successor_indices(idx)]

    def impacts(self, path: str, max_depth: int = 3) -> list[Ref]:
        """Reverse transitive reachability via BFS over predecessors.

        Returns nodes affected if *path* changes, up to *max_depth* hops.
        Excludes the starting node.  Cycle-safe.
        """
        idx = self._require_node(path)
        visited: set[int] = {idx}
        queue: deque[tuple[int, int]] = deque()
        for pred in self._graph.predecessor_indices(idx):
            if pred not in visited:
                queue.append((pred, 1))
                visited.add(pred)

        result: list[Ref] = []
        while queue:
            current, depth = queue.popleft()
            result.append(Ref(path=self._idx_to_path[current]))
            if depth < max_depth:
                for pred in self._graph.predecessor_indices(current):
                    if pred not in visited:
                        queue.append((pred, depth + 1))
                        visited.add(pred)
        return result

    def path_between(self, source: str, target: str) -> list[Ref] | None:
        """Shortest path (Dijkstra) from *source* to *target*, or ``None``."""
        src_idx = self._require_node(source)
        tgt_idx = self._require_node(target)
        if src_idx == tgt_idx:
            return [Ref(path=source)]
        try:
            paths = rustworkx.dijkstra_shortest_paths(
                self._graph,
                src_idx,
                target=tgt_idx,
                weight_fn=lambda e: e.get("weight", 1.0),
            )
            indices = paths[tgt_idx]
        except (KeyError, IndexError, rustworkx.NoPathFound):
            return None
        return [Ref(path=self._idx_to_path[i]) for i in indices]

    def contains(self, path: str) -> list[Ref]:
        """Successors connected by ``"contains"`` edges."""
        idx = self._require_node(path)
        result: list[Ref] = []
        for succ in self._graph.successor_indices(idx):
            edge_idx = self._find_edge_idx(idx, succ)
            if edge_idx is not None:
                data = self._graph.get_edge_data_by_index(edge_idx)
                if data.get("type") == "contains":
                    result.append(Ref(path=self._idx_to_path[succ]))
        return result

    def by_parent(self, parent_path: str) -> list[Ref]:
        """All nodes whose ``parent_path`` attribute matches."""
        result: list[Ref] = []
        for idx in self._graph.node_indices():
            data = self._graph[idx]
            if data.get("parent_path") == parent_path:
                result.append(Ref(path=data["path"]))
        return result

    def remove_file_subgraph(self, path: str) -> list[str]:
        """Remove a node and all nodes with ``parent_path == path``.

        Returns the list of removed paths.
        """
        self._require_node(path)
        # Collect child nodes first
        children = [
            self._idx_to_path[idx]
            for idx in self._graph.node_indices()
            if self._graph[idx].get("parent_path") == path
        ]
        removed = [path, *children]
        for p in removed:
            if self.has_node(p):
                self.remove_node(p)
        return removed

    # ------------------------------------------------------------------
    # Graph-level
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return self._graph.num_nodes()

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return self._graph.num_edges()

    def is_dag(self) -> bool:
        """Return whether the graph is a directed acyclic graph."""
        return rustworkx.is_directed_acyclic_graph(self._graph)

    def __repr__(self) -> str:
        return f"RustworkxGraph(nodes={self.node_count}, edges={self.edge_count})"

    # ------------------------------------------------------------------
    # Centrality algorithms (SupportsCentrality)
    # ------------------------------------------------------------------

    def pagerank(
        self,
        *,
        alpha: float = 0.85,
        personalization: dict[str, float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> dict[str, float]:
        """PageRank centrality scores."""
        pers = None
        if personalization:
            pers = {
                self._path_to_idx[p]: w
                for p, w in personalization.items()
                if p in self._path_to_idx
            }
            if not pers:
                pers = None
        scores = rustworkx.pagerank(
            self._graph, alpha=alpha, personalization=pers,
            max_iter=max_iter, tol=tol,
        )
        return {
            self._idx_to_path[idx]: score
            for idx, score in scores.items()
            if idx in self._idx_to_path
        }

    def betweenness_centrality(self, *, normalized: bool = True) -> dict[str, float]:
        """Betweenness centrality scores."""
        scores = rustworkx.digraph_betweenness_centrality(
            self._graph, normalized=normalized,
        )
        return {
            self._idx_to_path[idx]: score
            for idx, score in scores.items()
            if idx in self._idx_to_path
        }

    def closeness_centrality(self) -> dict[str, float]:
        """Closeness centrality scores."""
        scores = rustworkx.closeness_centrality(self._graph)
        return {
            self._idx_to_path[idx]: score
            for idx, score in scores.items()
            if idx in self._idx_to_path
        }

    def katz_centrality(
        self,
        *,
        alpha: float = 0.1,
        beta: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> dict[str, float]:
        """Katz centrality scores."""
        if self._graph.num_nodes() == 0:
            return {}
        scores = rustworkx.katz_centrality(
            self._graph, alpha=alpha, beta=beta,
            max_iter=max_iter, tol=tol,
        )
        return {
            self._idx_to_path[idx]: score
            for idx, score in scores.items()
            if idx in self._idx_to_path
        }

    def degree_centrality(self) -> dict[str, float]:
        """Degree centrality (in + out) scores."""
        scores = rustworkx.digraph_degree_centrality(self._graph)
        return {
            self._idx_to_path[idx]: score
            for idx, score in scores.items()
            if idx in self._idx_to_path
        }

    def in_degree_centrality(self) -> dict[str, float]:
        """In-degree centrality scores."""
        scores = rustworkx.in_degree_centrality(self._graph)
        return {
            self._idx_to_path[idx]: score
            for idx, score in scores.items()
            if idx in self._idx_to_path
        }

    def out_degree_centrality(self) -> dict[str, float]:
        """Out-degree centrality scores."""
        scores = rustworkx.out_degree_centrality(self._graph)
        return {
            self._idx_to_path[idx]: score
            for idx, score in scores.items()
            if idx in self._idx_to_path
        }

    # ------------------------------------------------------------------
    # Connectivity algorithms (SupportsConnectivity)
    # ------------------------------------------------------------------

    def weakly_connected_components(self) -> list[set[str]]:
        """Weakly connected components as sets of paths."""
        components = rustworkx.weakly_connected_components(self._graph)
        return [
            {self._idx_to_path[idx] for idx in comp if idx in self._idx_to_path}
            for comp in components
        ]

    def strongly_connected_components(self) -> list[set[str]]:
        """Strongly connected components as sets of paths."""
        components = rustworkx.strongly_connected_components(self._graph)
        return [
            {self._idx_to_path[idx] for idx in comp if idx in self._idx_to_path}
            for comp in components
        ]

    def is_weakly_connected(self) -> bool:
        """Return ``True`` if the graph is weakly connected.

        Returns ``True`` for an empty graph (matches convention that the
        null graph is trivially connected).
        """
        try:
            return rustworkx.is_weakly_connected(self._graph)
        except rustworkx.NullGraph:
            return True

    # ------------------------------------------------------------------
    # Traversal algorithms (SupportsTraversal)
    # ------------------------------------------------------------------

    def ancestors(self, path: str) -> set[str]:
        """All nodes reachable by following edges backward from *path*."""
        idx = self._require_node(path)
        return {
            self._idx_to_path[i]
            for i in rustworkx.ancestors(self._graph, idx)
            if i in self._idx_to_path
        }

    def descendants(self, path: str) -> set[str]:
        """All nodes reachable by following edges forward from *path*."""
        idx = self._require_node(path)
        return {
            self._idx_to_path[i]
            for i in rustworkx.descendants(self._graph, idx)
            if i in self._idx_to_path
        }

    def all_simple_paths(
        self, source: str, target: str, *, cutoff: int | None = None,
    ) -> list[list[str]]:
        """All simple (loop-free) paths from *source* to *target*.

        Parameters
        ----------
        cutoff:
            Maximum path length (number of nodes).  ``None`` means no limit.
        """
        src_idx = self._require_node(source)
        tgt_idx = self._require_node(target)
        raw = rustworkx.digraph_all_simple_paths(
            self._graph, src_idx, tgt_idx, cutoff=cutoff or 0,
        )
        return [
            [self._idx_to_path[i] for i in path]
            for path in raw
        ]

    def topological_sort(self) -> list[str]:
        """Return nodes in topological order.

        Raises ``ValueError`` if the graph contains cycles.
        """
        try:
            indices = rustworkx.topological_sort(self._graph)
        except rustworkx.DAGHasCycle:
            msg = "Graph contains cycles"
            raise ValueError(msg) from None
        return [self._idx_to_path[i] for i in indices if i in self._idx_to_path]

    def shortest_path_length(self, source: str, target: str) -> float | None:
        """Weighted shortest path length, or ``None`` if unreachable."""
        src_idx = self._require_node(source)
        tgt_idx = self._require_node(target)
        lengths = rustworkx.dijkstra_shortest_path_lengths(
            self._graph, src_idx,
            lambda e: e.get("weight", 1.0),
            goal=tgt_idx,
        )
        result = dict(lengths)
        return result.get(tgt_idx)

    # ------------------------------------------------------------------
    # Subgraph extraction (SupportsSubgraph)
    # ------------------------------------------------------------------

    def subgraph(self, paths: list[str]) -> SubgraphResult:
        """Extract the induced subgraph for the given *paths*.

        Missing paths are silently skipped.
        """
        valid = {p for p in paths if p in self._path_to_idx}
        edges: list[tuple[str, str, dict[str, Any]]] = []
        for src, tgt, data in self.edges():
            if src in valid and tgt in valid:
                edges.append((src, tgt, data))
        return subgraph_result(sorted(valid), edges)

    def neighborhood(
        self,
        path: str,
        *,
        max_depth: int = 2,
        direction: str = "both",
        edge_types: list[str] | None = None,
    ) -> SubgraphResult:
        """BFS neighborhood around *path* up to *max_depth* hops.

        Parameters
        ----------
        direction:
            ``"out"`` = successors only, ``"in"`` = predecessors only,
            ``"both"`` = both.
        edge_types:
            If provided, only traverse edges matching these types.
        """
        self._require_node(path)
        visited: set[str] = {path}
        frontier: set[str] = {path}

        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for node in frontier:
                idx = self._path_to_idx[node]
                neighbors: list[int] = []
                if direction in ("out", "both"):
                    neighbors.extend(self._graph.successor_indices(idx))
                if direction in ("in", "both"):
                    neighbors.extend(self._graph.predecessor_indices(idx))
                for n_idx in neighbors:
                    n_path = self._idx_to_path.get(n_idx)
                    if n_path is None or n_path in visited:
                        continue
                    # Edge type filter
                    if (
                        edge_types is not None
                        and not self._has_edge_of_type(idx, n_idx, edge_types, direction)
                    ):
                        continue
                    visited.add(n_path)
                    next_frontier.add(n_path)
            frontier = next_frontier
            if not frontier:
                break

        return self.subgraph(sorted(visited))

    def meeting_subgraph(
        self, start_paths: list[str], *, max_size: int = 50,
    ) -> SubgraphResult:
        """Find the subgraph connecting *start_paths* via shortest paths.

        Algorithm:
        1. Filter to valid start paths (need ≥ 2, else trivial subgraph).
        2. For each pair, find shortest path. Collect all intermediate nodes.
        3. If no pairwise connections, try common descendants.
        4. Score with PageRank (personalized on start nodes).
        5. Prune to *max_size* by removing lowest-score non-start nodes.
        """
        valid_starts = [p for p in start_paths if p in self._path_to_idx]
        if len(valid_starts) <= 1:
            return self.subgraph(valid_starts)

        # Step 1: collect all nodes on pairwise shortest paths
        all_nodes: set[str] = set(valid_starts)
        found_connection = False
        for i, src in enumerate(valid_starts):
            for tgt in valid_starts[i + 1 :]:
                path = self.path_between(src, tgt)
                if path is not None:
                    found_connection = True
                    for ref in path:
                        all_nodes.add(ref.path)
                # Also try reverse direction
                path_rev = self.path_between(tgt, src)
                if path_rev is not None:
                    found_connection = True
                    for ref in path_rev:
                        all_nodes.add(ref.path)

        # Step 2: if no pairwise connections, try common descendants
        if not found_connection:
            common = self.common_reachable(valid_starts, direction="forward")
            # Take up to 5 closest common nodes
            for node in list(common)[:5]:
                all_nodes.add(node)

        # Step 3: score with personalized PageRank
        pers = dict.fromkeys(valid_starts, 1.0)
        scores = self.pagerank(personalization=pers)

        # Step 4: prune to max_size
        start_set = set(valid_starts)
        node_list = sorted(all_nodes)
        while len(node_list) > max_size:
            # Find lowest-score non-start node
            worst: str | None = None
            worst_score = float("inf")
            for n in node_list:
                if n in start_set:
                    continue
                s = scores.get(n, 0.0)
                if s < worst_score:
                    worst_score = s
                    worst = n
            if worst is None:
                break
            node_list.remove(worst)

        # Build subgraph result with scores for included nodes
        sub = self.subgraph(node_list)
        sub_scores = {n: scores.get(n, 0.0) for n in sub.nodes}
        return subgraph_result(list(sub.nodes), list(sub.edges), sub_scores)

    def common_reachable(
        self, paths: list[str], *, direction: str = "forward",
    ) -> set[str]:
        """Intersection of descendants (forward) or ancestors (reverse)."""
        valid = [p for p in paths if p in self._path_to_idx]
        if not valid:
            return set()
        if direction == "forward":
            sets = [self.descendants(p) for p in valid]
        else:
            sets = [self.ancestors(p) for p in valid]
        result = sets[0]
        for s in sets[1:]:
            result = result & s
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def to_sql(self, session: AsyncSession) -> None:
        """Persist the graph to ``grover_edges``.

        Full-sync strategy: upsert all current edges, delete DB edges that are
        no longer present in the in-memory graph.  Caller manages the
        transaction (commit/rollback).
        """
        from sqlalchemy import select

        from grover.models.edges import GroverEdge

        # Build GroverEdge instances from in-memory graph
        graph_edges: dict[str, GroverEdge] = {}
        for _src, _tgt, data in self.edges():
            edge = GroverEdge(
                id=data["id"],
                source_path=data["source"],
                target_path=data["target"],
                type=data["type"],
                weight=data["weight"],
                metadata_json=json.dumps(data["metadata"]),
            )
            graph_edges[edge.id] = edge

        # Find existing DB edge IDs
        result = await session.execute(select(GroverEdge.id))  # type: ignore[no-matching-overload]
        db_ids: set[str] = {row[0] for row in result.all()}

        # Delete stale edges
        graph_ids = set(graph_edges.keys())
        stale_ids = db_ids - graph_ids
        if stale_ids:
            for stale_id in stale_ids:
                existing = await session.get(GroverEdge, stale_id)
                if existing:
                    await session.delete(existing)

        # Upsert current edges
        for edge in graph_edges.values():
            await session.merge(edge)

        await session.flush()

    async def from_sql(self, session: AsyncSession, file_model: type | None = None) -> None:
        """Load graph state from the database, replacing in-memory state.

        Loads non-deleted file rows as nodes and all ``GroverEdge``
        rows as edges.  Auto-creates nodes for dangling edge endpoints.

        Parameters
        ----------
        session:
            An async SQLAlchemy session.
        file_model:
            The SQLModel class to query for file nodes.  Defaults to
            :class:`~grover.models.files.File`.
        """
        from sqlalchemy import select

        from grover.models.edges import GroverEdge

        if file_model is None:
            from grover.models.files import File

            file_model = File

        # Reset
        self._graph = rustworkx.PyDiGraph()
        self._path_to_idx = {}
        self._idx_to_path = {}

        # Load non-deleted files as nodes
        result = await session.execute(
            select(file_model).where(file_model.deleted_at.is_(None))  # type: ignore[union-attr]
        )
        for file_row in result.scalars().all():
            self.add_node(
                file_row.path,  # type: ignore[unresolved-attribute]
                parent_path=file_row.parent_path,  # type: ignore[unresolved-attribute]
                is_directory=file_row.is_directory,  # type: ignore[unresolved-attribute]
            )

        # Load all edges
        result = await session.execute(select(GroverEdge))
        for edge_row in result.scalars().all():
            metadata: dict[str, Any] = json.loads(edge_row.metadata_json)
            self.add_edge(
                edge_row.source_path,
                edge_row.target_path,
                edge_row.type,
                weight=edge_row.weight,
                edge_id=edge_row.id,
                **metadata,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_node(self, path: str) -> int:
        """Return the rustworkx index for *path*, or raise ``KeyError``."""
        try:
            return self._path_to_idx[path]
        except KeyError:
            msg = f"Node not found: {path!r}"
            raise KeyError(msg) from None

    def _find_edge_idx(self, src_idx: int, tgt_idx: int) -> int | None:
        """Return the edge index between two node indices, or ``None``."""
        try:
            indices = self._graph.edge_indices_from_endpoints(src_idx, tgt_idx)
        except Exception:
            return None
        return indices[0] if indices else None

    def _has_edge_of_type(
        self, node_idx: int, neighbor_idx: int,
        edge_types: list[str], direction: str,
    ) -> bool:
        """Check if there's an edge of one of *edge_types* between node and neighbor."""
        pairs: list[tuple[int, int]] = []
        if direction in ("out", "both"):
            pairs.append((node_idx, neighbor_idx))
        if direction in ("in", "both"):
            pairs.append((neighbor_idx, node_idx))
        for src, tgt in pairs:
            edge_idx = self._find_edge_idx(src, tgt)
            if edge_idx is not None:
                data = self._graph.get_edge_data_by_index(edge_idx)
                if data.get("type") in edge_types:
                    return True
        return False
