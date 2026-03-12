"""RustworkxGraph — rustworkx-backed graph store implementing GraphProvider protocol."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any

import rustworkx

from grover.ref import Ref
from grover.results.search import (
    AncestorsResult,
    BetweennessResult,
    ClosenessResult,
    CommonNeighborsResult,
    ConnectionCandidate,
    DegreeResult,
    DescendantsResult,
    EgoGraphResult,
    FileCandidate,
    GraphEvidence,
    HarmonicResult,
    HasPathResult,
    HitsResult,
    KatzResult,
    MeetingSubgraphResult,
    PageRankResult,
    PredecessorsResult,
    ShortestPathResult,
    SubgraphSearchResult,
    SuccessorsResult,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.results.search import FileSearchResult


class RustworkxGraph:
    """Directed knowledge graph over file paths.

    Stores only topology as pure Python sets (``_nodes``, ``_edges``).
    Query/algorithm methods are ``async def``:
    - Light reads run inline (no thread overhead).
    - Heavy algorithms use ``asyncio.to_thread`` with a snapshot for concurrency.

    Mutations stay synchronous (trivial set operations, called from background tasks).

    Implements the ``GraphProvider`` protocol.
    """

    def __init__(self, *, stale_after: float | None = None) -> None:
        self._nodes: set[str] = set()
        self._edges: set[tuple[str, str]] = set()
        # Staleness tracking
        self._loaded_at: float | None = None
        self._stale_after: float | None = stale_after
        # Refresh config (set via configure_refresh)
        self._refresh_file_model: type | None = None
        self._refresh_path_prefix: str = ""

    # ------------------------------------------------------------------
    # Staleness tracking and self-refresh
    # ------------------------------------------------------------------

    @property
    def needs_refresh(self) -> bool:
        """True if the graph has never been loaded or its TTL has expired.

        A graph that has been populated through mutations (add_node/add_edge)
        is considered initialized even without ``from_sql()``, so it won't
        trigger an unwanted reload that would wipe in-memory-only edges.
        """
        if self._loaded_at is None:
            # Never loaded from SQL — only refresh if the graph is also empty.
            # A non-empty graph was populated via writes (warm from mutations).
            return not self._nodes
        if self._stale_after is None:
            return False  # No TTL — manual only
        return (time.monotonic() - self._loaded_at) > self._stale_after

    @property
    def stale_after(self) -> float | None:
        """TTL in seconds, or ``None`` for no automatic refresh."""
        return self._stale_after

    @stale_after.setter
    def stale_after(self, value: float | None) -> None:
        self._stale_after = value

    @property
    def loaded_at(self) -> float | None:
        """Monotonic timestamp of the last ``from_sql()`` load, or ``None``."""
        return self._loaded_at

    def configure_refresh(
        self,
        file_model: type | None = None,
        path_prefix: str = "",
    ) -> None:
        """Store refresh parameters so ``_ensure_fresh`` can call ``from_sql``."""
        self._refresh_file_model = file_model
        self._refresh_path_prefix = path_prefix

    async def _ensure_fresh(self, session: AsyncSession | None) -> None:
        """Load from DB if never loaded or TTL exceeded."""
        if not self.needs_refresh:
            return
        if session is None:
            return  # No session available — serve from memory as-is
        await self.from_sql(
            session,
            self._refresh_file_model,
            path_prefix=self._refresh_path_prefix,
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _snapshot(self) -> tuple[frozenset[str], frozenset[tuple[str, str]]]:
        """Return immutable copies of nodes and edges for thread-safe reads."""
        return frozenset(self._nodes), frozenset(self._edges)

    @staticmethod
    def _build_graph_from(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
    ) -> tuple[rustworkx.PyDiGraph, dict[str, int], dict[int, str]]:
        """Build a PyDiGraph from explicit node/edge sets."""
        graph: rustworkx.PyDiGraph = rustworkx.PyDiGraph()
        path_to_idx: dict[str, int] = {}
        idx_to_path: dict[int, str] = {}
        for path in nodes:
            idx = graph.add_node(path)
            path_to_idx[path] = idx
            idx_to_path[idx] = path
        for source, target in edges:
            if source in path_to_idx and target in path_to_idx:
                graph.add_edge(path_to_idx[source], path_to_idx[target], None)
        return graph, path_to_idx, idx_to_path

    def _build_graph(self) -> tuple[rustworkx.PyDiGraph, dict[str, int], dict[int, str]]:
        """Build a fresh PyDiGraph from current _nodes/_edges."""
        return self._build_graph_from(frozenset(self._nodes), frozenset(self._edges))

    # ------------------------------------------------------------------
    # Node operations (sync mutations)
    # ------------------------------------------------------------------

    def add_node(self, path: str, **attrs: object) -> None:
        """Add a node. Extra *attrs* are accepted for protocol compat but not stored."""
        if path not in self._nodes:
            self._nodes.add(path)

    def remove_node(self, path: str) -> None:
        """Remove a node and all incident edges. Raises ``KeyError`` if missing."""
        if path not in self._nodes:
            msg = f"Node not found: {path!r}"
            raise KeyError(msg)
        self._nodes.discard(path)
        self._edges = {(s, t) for s, t in self._edges if s != path and t != path}

    def has_node(self, path: str) -> bool:
        return path in self._nodes

    def get_node(self, path: str) -> dict[str, Any]:
        """Return minimal node data dict. Raises ``KeyError`` if missing."""
        if path not in self._nodes:
            msg = f"Node not found: {path!r}"
            raise KeyError(msg)
        return {"path": path}

    def nodes(self) -> list[str]:
        return list(self._nodes)

    # ------------------------------------------------------------------
    # Edge operations (sync mutations)
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        *,
        weight: float = 1.0,
        edge_id: str | None = None,
        **attrs: object,
    ) -> None:
        """Add a directed edge. Auto-creates missing endpoint nodes."""
        self._nodes.add(source)
        self._nodes.add(target)
        self._edges.add((source, target))

    def remove_edge(self, source: str, target: str) -> None:
        """Remove the edge between *source* and *target*. Raises ``KeyError``."""
        if source not in self._nodes:
            msg = f"Node not found: {source!r}"
            raise KeyError(msg)
        if target not in self._nodes:
            msg = f"Node not found: {target!r}"
            raise KeyError(msg)
        if (source, target) not in self._edges:
            msg = f"No edge from {source!r} to {target!r}"
            raise KeyError(msg)
        self._edges.discard((source, target))

    def has_edge(self, source: str, target: str) -> bool:
        return (source, target) in self._edges

    def get_edge(self, source: str, target: str) -> dict[str, Any]:
        """Return minimal edge data dict. Raises ``KeyError`` if missing."""
        if source not in self._nodes:
            msg = f"Node not found: {source!r}"
            raise KeyError(msg)
        if target not in self._nodes:
            msg = f"Node not found: {target!r}"
            raise KeyError(msg)
        if (source, target) not in self._edges:
            msg = f"No edge from {source!r} to {target!r}"
            raise KeyError(msg)
        return {
            "id": str(uuid.uuid4()),
            "source": source,
            "target": target,
            "type": "",
            "weight": 1.0,
            "metadata": {},
        }

    def edges(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Return all edges as ``(source, target, data)`` triples."""
        return [
            (s, t, {"id": "", "source": s, "target": t, "type": "", "weight": 1.0, "metadata": {}})
            for s, t in self._edges
        ]

    # ------------------------------------------------------------------
    # Public property: access the underlying rustworkx graph
    # ------------------------------------------------------------------

    @property
    def graph(self) -> rustworkx.PyDiGraph:
        """Access the underlying rustworkx directed graph."""
        g, _, _ = self._build_graph_from(frozenset(self._nodes), frozenset(self._edges))
        return g

    # ------------------------------------------------------------------
    # Light reads — async inline (no thread overhead)
    # ------------------------------------------------------------------

    async def predecessors(
        self, path: str, *, session: AsyncSession | None = None
    ) -> PredecessorsResult:
        """Nodes with edges pointing *to* this node."""
        await self._ensure_fresh(session)
        self._require_node(path)
        preds = sorted({s for s, t in self._edges if t == path})
        return PredecessorsResult(
            success=True,
            message=f"{len(preds)} predecessor(s)",
            file_candidates=[
                FileCandidate(path=p, evidence=[GraphEvidence(operation="predecessors")])
                for p in preds
            ],
        )

    async def successors(
        self, path: str, *, session: AsyncSession | None = None
    ) -> SuccessorsResult:
        """Nodes this node points *to*."""
        await self._ensure_fresh(session)
        self._require_node(path)
        succs = sorted({t for s, t in self._edges if s == path})
        return SuccessorsResult(
            success=True,
            message=f"{len(succs)} successor(s)",
            file_candidates=[
                FileCandidate(path=p, evidence=[GraphEvidence(operation="successors")])
                for p in succs
            ],
        )

    async def contains(self, path: str) -> list[Ref]:
        """Successors as Refs (internal use — not part of typed-result API)."""
        self._require_node(path)
        return [Ref(path=t) for s, t in self._edges if s == path]

    async def by_parent(self, parent_path: str) -> list[Ref]:
        """Not supported with minimal storage — returns empty list."""
        return []

    async def subgraph(
        self, paths: list[str], *, session: AsyncSession | None = None
    ) -> SubgraphSearchResult:
        """Extract the induced subgraph for the given *paths*."""
        await self._ensure_fresh(session)
        valid = {p for p in paths if p in self._nodes}
        edge_list: list[tuple[str, str, dict[str, Any]]] = []
        for s, t in self._edges:
            if s in valid and t in valid:
                edge_list.append((s, t, {"type": "", "weight": 1.0}))
        nodes_sorted = sorted(valid)
        return SubgraphSearchResult(
            success=True,
            message=f"{len(nodes_sorted)} node(s), {len(edge_list)} edge(s)",
            file_candidates=[
                FileCandidate(path=n, evidence=[GraphEvidence(operation="subgraph")])
                for n in nodes_sorted
            ],
            connection_candidates=[
                ConnectionCandidate(
                    source_path=s,
                    target_path=t,
                    connection_type=data.get("type", ""),
                    weight=data.get("weight", 1.0),
                    evidence=[GraphEvidence(operation="subgraph")],
                )
                for s, t, data in edge_list
            ],
        )

    async def neighborhood(
        self,
        path: str,
        *,
        max_depth: int = 2,
        direction: str = "both",
        edge_types: list[str] | None = None,
        session: AsyncSession | None = None,
    ) -> EgoGraphResult:
        """BFS neighborhood around *path* up to *max_depth* hops."""
        await self._ensure_fresh(session)
        self._require_node(path)
        out_adj: dict[str, set[str]] = {}
        in_adj: dict[str, set[str]] = {}
        for s, t in self._edges:
            out_adj.setdefault(s, set()).add(t)
            in_adj.setdefault(t, set()).add(s)

        visited: set[str] = {path}
        frontier: set[str] = {path}

        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for node in frontier:
                neighbors: set[str] = set()
                if direction in ("out", "both"):
                    neighbors |= out_adj.get(node, set())
                if direction in ("in", "both"):
                    neighbors |= in_adj.get(node, set())
                for n in neighbors:
                    if n not in visited:
                        visited.add(n)
                        next_frontier.add(n)
            frontier = next_frontier
            if not frontier:
                break

        sub = await self.subgraph(sorted(visited), session=session)
        return EgoGraphResult(
            success=sub.success,
            message=sub.message,
            file_candidates=sub.file_candidates,
            connection_candidates=sub.connection_candidates,
        )

    async def common_neighbors(
        self, path1: str, path2: str, *, session: AsyncSession | None = None
    ) -> CommonNeighborsResult:
        """Intersection of undirected neighbors of both nodes."""
        await self._ensure_fresh(session)
        neighbors = sorted(self._undirected_neighbors(path1) & self._undirected_neighbors(path2))
        return CommonNeighborsResult(
            success=True,
            message=f"{len(neighbors)} common neighbor(s)",
            file_candidates=[
                FileCandidate(path=p, evidence=[GraphEvidence(operation="common_neighbors")])
                for p in neighbors
            ],
        )

    async def connecting_subgraph(
        self, paths: list[str], *, session: AsyncSession | None = None
    ) -> RustworkxGraph:
        """Return a new RustworkxGraph containing all nodes needed to connect *paths*."""
        await self._ensure_fresh(session)
        valid = [p for p in paths if p in self._nodes]
        if len(valid) <= 1:
            sub = RustworkxGraph()
            sub._nodes = set(valid)
            sub._edges = set()
            return sub

        graph, path_to_idx, idx_to_path = self._build_graph()
        seed_indices = [path_to_idx[p] for p in valid]
        keep_indices = self._multisource_bfs(graph, seed_indices)
        keep_paths = {idx_to_path[i] for i in keep_indices if i in idx_to_path}

        sub = RustworkxGraph()
        sub._nodes = keep_paths
        sub._edges = {(s, t) for s, t in self._edges if s in keep_paths and t in keep_paths}
        return sub

    async def node_similarity(
        self,
        path1: str,
        path2: str,
        *,
        method: str = "jaccard",
        session: AsyncSession | None = None,
    ) -> float:
        await self._ensure_fresh(session)
        n1 = self._undirected_neighbors(path1)
        n2 = self._undirected_neighbors(path2)
        union = n1 | n2
        if not union:
            return 0.0
        return len(n1 & n2) / len(union)

    async def similar_nodes(
        self,
        path: str,
        *,
        method: str = "jaccard",
        k: int = 10,
        session: AsyncSession | None = None,
    ) -> list[tuple[str, float]]:
        await self._ensure_fresh(session)
        scores: list[tuple[str, float]] = []
        for other in self._nodes:
            if other == path:
                continue
            s = await self.node_similarity(path, other, method=method, session=session)
            scores.append((other, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    # ------------------------------------------------------------------
    # Sync utilities
    # ------------------------------------------------------------------

    def remove_file_subgraph(self, path: str) -> list[str]:
        """Remove a node and all its successors connected by any edge."""
        if path not in self._nodes:
            msg = f"Node not found: {path!r}"
            raise KeyError(msg)
        children = {t for s, t in self._edges if s == path}
        removed = [path, *sorted(children)]
        for p in removed:
            if p in self._nodes:
                self._nodes.discard(p)
                self._edges = {(s, t) for s, t in self._edges if s != p and t != p}
        return removed

    # ------------------------------------------------------------------
    # Graph-level (sync properties/utilities)
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def is_dag(self) -> bool:
        graph, _, _ = self._build_graph()
        return rustworkx.is_directed_acyclic_graph(graph)

    def __repr__(self) -> str:
        return f"RustworkxGraph(nodes={self.node_count}, edges={self.edge_count})"

    # ------------------------------------------------------------------
    # Heavy algorithms — async + to_thread with snapshot
    # ------------------------------------------------------------------

    @staticmethod
    def _scores_to_candidates(
        scores: dict[str, float], operation: str, algorithm: str
    ) -> list[FileCandidate]:
        """Convert a {path: score} dict to sorted FileCandidate list."""
        sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            FileCandidate(
                path=path,
                evidence=[GraphEvidence(operation=operation, algorithm=algorithm, score=score)],
            )
            for path, score in sorted_items
        ]

    # --- resolve_graph_from: thread-safe version of _resolve_graph ---

    @staticmethod
    def _resolve_graph_from(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
        candidates: FileSearchResult | None,
    ) -> tuple[rustworkx.PyDiGraph, dict[str, int], dict[int, str]]:
        """Thread-safe resolve: build the appropriate graph from snapshot data."""
        if candidates is None:
            return RustworkxGraph._build_graph_from(nodes, edges)
        if candidates.connection_candidates:
            sub_nodes = frozenset(c.path for c in candidates.file_candidates)
            sub_edges = frozenset(
                (cc.source_path, cc.target_path) for cc in candidates.connection_candidates
            )
            return RustworkxGraph._build_graph_from(sub_nodes, sub_edges)
        paths = [c.path for c in candidates.file_candidates]
        if not paths:
            return RustworkxGraph._build_graph_from(nodes, edges)
        # Build connecting subgraph
        valid = [p for p in paths if p in nodes]
        if len(valid) <= 1:
            sub_nodes = frozenset(valid)
            sub_edges = frozenset((s, t) for s, t in edges if s in sub_nodes and t in sub_nodes)
            return RustworkxGraph._build_graph_from(sub_nodes, sub_edges)
        # Full connecting subgraph via multisource BFS
        graph, path_to_idx, idx_to_path = RustworkxGraph._build_graph_from(nodes, edges)
        seed_indices = [path_to_idx[p] for p in valid]
        keep_indices = RustworkxGraph._multisource_bfs_static(graph, seed_indices)
        keep_paths = frozenset(idx_to_path[i] for i in keep_indices if i in idx_to_path)
        keep_edges = frozenset((s, t) for s, t in edges if s in keep_paths and t in keep_paths)
        return RustworkxGraph._build_graph_from(keep_paths, keep_edges)

    # --- PageRank ---

    async def pagerank(
        self,
        candidates: FileSearchResult | None = None,
        *,
        alpha: float = 0.85,
        personalization: dict[str, float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        session: AsyncSession | None = None,
    ) -> PageRankResult:
        """PageRank centrality scores."""
        await self._ensure_fresh(session)
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(
            self._pagerank_impl, nodes, edges, candidates, alpha, personalization, max_iter, tol
        )

    @staticmethod
    def _pagerank_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
        candidates: FileSearchResult | None,
        alpha: float,
        personalization: dict[str, float] | None,
        max_iter: int,
        tol: float,
    ) -> PageRankResult:
        graph, path_to_idx, idx_to_path = RustworkxGraph._resolve_graph_from(
            nodes, edges, candidates
        )
        if graph.num_nodes() == 0:
            return PageRankResult(success=True, message="0 node(s)")
        pers = None
        if personalization:
            pers = {path_to_idx[p]: w for p, w in personalization.items() if p in path_to_idx}
            if not pers:
                pers = None
        scores = rustworkx.pagerank(
            graph, alpha=alpha, personalization=pers, max_iter=max_iter, tol=tol
        )
        raw = {idx_to_path[idx]: score for idx, score in scores.items() if idx in idx_to_path}
        fcs = RustworkxGraph._scores_to_candidates(raw, "pagerank", "pagerank")
        return PageRankResult(
            success=True,
            message=f"{len(fcs)} node(s)",
            file_candidates=fcs,
        )

    # --- Betweenness ---

    async def betweenness_centrality(
        self,
        candidates: FileSearchResult | None = None,
        *,
        normalized: bool = True,
        session: AsyncSession | None = None,
    ) -> BetweennessResult:
        """Betweenness centrality scores."""
        await self._ensure_fresh(session)
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(self._betweenness_impl, nodes, edges, candidates, normalized)

    @staticmethod
    def _betweenness_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
        candidates: FileSearchResult | None,
        normalized: bool,
    ) -> BetweennessResult:
        graph, _, idx_to_path = RustworkxGraph._resolve_graph_from(nodes, edges, candidates)
        if graph.num_nodes() == 0:
            return BetweennessResult(success=True, message="0 node(s)")
        scores = rustworkx.digraph_betweenness_centrality(graph, normalized=normalized)
        raw = {idx_to_path[idx]: score for idx, score in scores.items() if idx in idx_to_path}
        fcs = RustworkxGraph._scores_to_candidates(
            raw, "betweenness_centrality", "betweenness_centrality"
        )
        return BetweennessResult(
            success=True,
            message=f"{len(fcs)} node(s)",
            file_candidates=fcs,
        )

    # --- Closeness ---

    async def closeness_centrality(
        self,
        candidates: FileSearchResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> ClosenessResult:
        """Closeness centrality scores."""
        await self._ensure_fresh(session)
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(self._closeness_impl, nodes, edges, candidates)

    @staticmethod
    def _closeness_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
        candidates: FileSearchResult | None,
    ) -> ClosenessResult:
        graph, _, idx_to_path = RustworkxGraph._resolve_graph_from(nodes, edges, candidates)
        if graph.num_nodes() == 0:
            return ClosenessResult(success=True, message="0 node(s)")
        scores = rustworkx.closeness_centrality(graph)
        raw = {idx_to_path[idx]: score for idx, score in scores.items() if idx in idx_to_path}
        fcs = RustworkxGraph._scores_to_candidates(
            raw, "closeness_centrality", "closeness_centrality"
        )
        return ClosenessResult(
            success=True,
            message=f"{len(fcs)} node(s)",
            file_candidates=fcs,
        )

    # --- Harmonic ---

    async def harmonic_centrality(
        self,
        candidates: FileSearchResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> HarmonicResult:
        """Harmonic centrality scores."""
        await self._ensure_fresh(session)
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(self._harmonic_impl, nodes, edges, candidates)

    @staticmethod
    def _harmonic_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
        candidates: FileSearchResult | None,
    ) -> HarmonicResult:
        graph, path_to_idx, _ = RustworkxGraph._resolve_graph_from(nodes, edges, candidates)
        raw: dict[str, float] = {}
        for path, idx in path_to_idx.items():
            lengths = rustworkx.dijkstra_shortest_path_lengths(graph, idx, lambda _e: 1.0)
            score = sum(1.0 / d for d in dict(lengths).values() if d > 0)
            raw[path] = score
        fcs = RustworkxGraph._scores_to_candidates(
            raw, "harmonic_centrality", "harmonic_centrality"
        )
        return HarmonicResult(
            success=True,
            message=f"{len(fcs)} node(s)",
            file_candidates=fcs,
        )

    # --- HITS ---

    async def hits(
        self,
        candidates: FileSearchResult | None = None,
        *,
        max_iter: int = 100,
        tol: float = 1e-8,
        session: AsyncSession | None = None,
    ) -> HitsResult:
        """HITS hub and authority scores."""
        await self._ensure_fresh(session)
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(self._hits_impl, nodes, edges, candidates, max_iter, tol)

    @staticmethod
    def _hits_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
        candidates: FileSearchResult | None,
        max_iter: int,
        tol: float,
    ) -> HitsResult:
        graph, path_to_idx, idx_to_path = RustworkxGraph._resolve_graph_from(
            nodes, edges, candidates
        )
        if graph.num_nodes() == 0 or graph.num_edges() == 0:
            all_paths = sorted(path_to_idx)
            return HitsResult(
                success=True,
                message=f"HITS computed for {len(all_paths)} node(s)",
                file_candidates=[
                    FileCandidate(
                        path=p,
                        evidence=[
                            GraphEvidence(operation="hits_authority", algorithm="hits", score=0.0),
                            GraphEvidence(operation="hits_hub", algorithm="hits", score=0.0),
                        ],
                    )
                    for p in all_paths
                ],
            )
        hubs_raw, auths_raw = rustworkx.hits(graph, max_iter=max_iter, tol=tol)
        hubs = {idx_to_path[idx]: score for idx, score in hubs_raw.items() if idx in idx_to_path}
        auths = {idx_to_path[idx]: score for idx, score in auths_raw.items() if idx in idx_to_path}
        all_paths = sorted(set(hubs) | set(auths), key=lambda p: auths.get(p, 0.0), reverse=True)
        return HitsResult(
            success=True,
            message=f"HITS computed for {len(all_paths)} node(s)",
            file_candidates=[
                FileCandidate(
                    path=p,
                    evidence=[
                        GraphEvidence(
                            operation="hits_authority", algorithm="hits", score=auths.get(p, 0.0)
                        ),
                        GraphEvidence(
                            operation="hits_hub", algorithm="hits", score=hubs.get(p, 0.0)
                        ),
                    ],
                )
                for p in all_paths
            ],
        )

    # --- Katz ---

    async def katz_centrality(
        self,
        candidates: FileSearchResult | None = None,
        *,
        alpha: float = 0.1,
        beta: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
        session: AsyncSession | None = None,
    ) -> KatzResult:
        """Katz centrality scores."""
        await self._ensure_fresh(session)
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(
            self._katz_impl, nodes, edges, candidates, alpha, beta, max_iter, tol
        )

    @staticmethod
    def _katz_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
        candidates: FileSearchResult | None,
        alpha: float,
        beta: float,
        max_iter: int,
        tol: float,
    ) -> KatzResult:
        graph, _, idx_to_path = RustworkxGraph._resolve_graph_from(nodes, edges, candidates)
        if graph.num_nodes() == 0:
            return KatzResult(success=True, message="0 node(s)")
        scores = rustworkx.katz_centrality(
            graph, alpha=alpha, beta=beta, max_iter=max_iter, tol=tol
        )
        raw = {idx_to_path[idx]: score for idx, score in scores.items() if idx in idx_to_path}
        fcs = RustworkxGraph._scores_to_candidates(raw, "katz_centrality", "katz_centrality")
        return KatzResult(
            success=True,
            message=f"{len(fcs)} node(s)",
            file_candidates=fcs,
        )

    # --- Degree centrality ---

    async def degree_centrality(
        self,
        candidates: FileSearchResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> DegreeResult:
        """Degree centrality (in + out) scores."""
        await self._ensure_fresh(session)
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(self._degree_impl, nodes, edges, candidates)

    @staticmethod
    def _degree_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
        candidates: FileSearchResult | None,
    ) -> DegreeResult:
        graph, _, idx_to_path = RustworkxGraph._resolve_graph_from(nodes, edges, candidates)
        if graph.num_nodes() == 0:
            return DegreeResult(success=True, message="0 node(s)")
        scores = rustworkx.digraph_degree_centrality(graph)
        raw = {idx_to_path[idx]: score for idx, score in scores.items() if idx in idx_to_path}
        fcs = RustworkxGraph._scores_to_candidates(raw, "degree_centrality", "degree_centrality")
        return DegreeResult(
            success=True,
            message=f"{len(fcs)} node(s)",
            file_candidates=fcs,
        )

    async def in_degree_centrality(
        self,
        candidates: FileSearchResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> DegreeResult:
        """In-degree centrality scores."""
        await self._ensure_fresh(session)
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(self._in_degree_impl, nodes, edges, candidates)

    @staticmethod
    def _in_degree_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
        candidates: FileSearchResult | None,
    ) -> DegreeResult:
        graph, _, idx_to_path = RustworkxGraph._resolve_graph_from(nodes, edges, candidates)
        if graph.num_nodes() == 0:
            return DegreeResult(success=True, message="0 node(s)")
        scores = rustworkx.in_degree_centrality(graph)
        raw = {idx_to_path[idx]: score for idx, score in scores.items() if idx in idx_to_path}
        fcs = RustworkxGraph._scores_to_candidates(
            raw, "in_degree_centrality", "in_degree_centrality"
        )
        return DegreeResult(
            success=True,
            message=f"{len(fcs)} node(s)",
            file_candidates=fcs,
        )

    async def out_degree_centrality(
        self,
        candidates: FileSearchResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> DegreeResult:
        """Out-degree centrality scores."""
        await self._ensure_fresh(session)
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(self._out_degree_impl, nodes, edges, candidates)

    @staticmethod
    def _out_degree_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
        candidates: FileSearchResult | None,
    ) -> DegreeResult:
        graph, _, idx_to_path = RustworkxGraph._resolve_graph_from(nodes, edges, candidates)
        if graph.num_nodes() == 0:
            return DegreeResult(success=True, message="0 node(s)")
        scores = rustworkx.out_degree_centrality(graph)
        raw = {idx_to_path[idx]: score for idx, score in scores.items() if idx in idx_to_path}
        fcs = RustworkxGraph._scores_to_candidates(
            raw, "out_degree_centrality", "out_degree_centrality"
        )
        return DegreeResult(
            success=True,
            message=f"{len(fcs)} node(s)",
            file_candidates=fcs,
        )

    # --- Connectivity ---

    async def weakly_connected_components(
        self, *, session: AsyncSession | None = None
    ) -> list[set[str]]:
        await self._ensure_fresh(session)
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(self._weakly_connected_impl, nodes, edges)

    @staticmethod
    def _weakly_connected_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
    ) -> list[set[str]]:
        graph, _, idx_to_path = RustworkxGraph._build_graph_from(nodes, edges)
        components = rustworkx.weakly_connected_components(graph)
        return [{idx_to_path[idx] for idx in comp if idx in idx_to_path} for comp in components]

    async def strongly_connected_components(
        self, *, session: AsyncSession | None = None
    ) -> list[set[str]]:
        await self._ensure_fresh(session)
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(self._strongly_connected_impl, nodes, edges)

    @staticmethod
    def _strongly_connected_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
    ) -> list[set[str]]:
        graph, _, idx_to_path = RustworkxGraph._build_graph_from(nodes, edges)
        components = rustworkx.strongly_connected_components(graph)
        return [{idx_to_path[idx] for idx in comp if idx in idx_to_path} for comp in components]

    async def is_weakly_connected(self, *, session: AsyncSession | None = None) -> bool:
        await self._ensure_fresh(session)
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(self._is_weakly_connected_impl, nodes, edges)

    @staticmethod
    def _is_weakly_connected_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
    ) -> bool:
        graph, _, _ = RustworkxGraph._build_graph_from(nodes, edges)
        try:
            return rustworkx.is_weakly_connected(graph)
        except rustworkx.NullGraph:
            return True

    # --- Traversal ---

    async def ancestors(self, path: str, *, session: AsyncSession | None = None) -> AncestorsResult:
        await self._ensure_fresh(session)
        self._require_node(path)
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(self._ancestors_impl, nodes, edges, path)

    @staticmethod
    def _ancestors_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
        path: str,
    ) -> AncestorsResult:
        graph, path_to_idx, idx_to_path = RustworkxGraph._build_graph_from(nodes, edges)
        idx = path_to_idx[path]
        result = sorted(idx_to_path[i] for i in rustworkx.ancestors(graph, idx) if i in idx_to_path)
        return AncestorsResult(
            success=True,
            message=f"{len(result)} ancestor(s)",
            file_candidates=[
                FileCandidate(path=p, evidence=[GraphEvidence(operation="ancestors")])
                for p in result
            ],
        )

    async def descendants(
        self, path: str, *, session: AsyncSession | None = None
    ) -> DescendantsResult:
        await self._ensure_fresh(session)
        self._require_node(path)
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(self._descendants_impl, nodes, edges, path)

    @staticmethod
    def _descendants_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
        path: str,
    ) -> DescendantsResult:
        graph, path_to_idx, idx_to_path = RustworkxGraph._build_graph_from(nodes, edges)
        idx = path_to_idx[path]
        result = sorted(
            idx_to_path[i] for i in rustworkx.descendants(graph, idx) if i in idx_to_path
        )
        return DescendantsResult(
            success=True,
            message=f"{len(result)} descendant(s)",
            file_candidates=[
                FileCandidate(path=p, evidence=[GraphEvidence(operation="descendants")])
                for p in result
            ],
        )

    async def path_between(
        self, source: str, target: str, *, session: AsyncSession | None = None
    ) -> ShortestPathResult:
        """Shortest path (Dijkstra) from *source* to *target*."""
        await self._ensure_fresh(session)
        self._require_node(source)
        self._require_node(target)
        if source == target:
            return ShortestPathResult(
                success=True,
                message="Path of 1 node(s)",
                file_candidates=[
                    FileCandidate(
                        path=source,
                        evidence=[GraphEvidence(operation="shortest_path")],
                    )
                ],
            )
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(self._path_between_impl, nodes, edges, source, target)

    @staticmethod
    def _path_between_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
        source: str,
        target: str,
    ) -> ShortestPathResult:
        graph, path_to_idx, idx_to_path = RustworkxGraph._build_graph_from(nodes, edges)
        src_idx = path_to_idx[source]
        tgt_idx = path_to_idx[target]
        try:
            paths = rustworkx.dijkstra_shortest_paths(
                graph, src_idx, target=tgt_idx, weight_fn=lambda _e: 1.0
            )
            indices = paths[tgt_idx]
        except (KeyError, IndexError, rustworkx.NoPathFound):
            return ShortestPathResult(success=True, message="No path found")
        node_paths = [idx_to_path[i] for i in indices]
        return ShortestPathResult(
            success=True,
            message=f"Path of {len(node_paths)} node(s)",
            file_candidates=[
                FileCandidate(path=p, evidence=[GraphEvidence(operation="shortest_path")])
                for p in node_paths
            ],
        )

    async def has_path(
        self, source: str, target: str, *, session: AsyncSession | None = None
    ) -> HasPathResult:
        result = await self.path_between(source, target, session=session)
        if not result:
            return HasPathResult(success=True, message="No path exists")
        return HasPathResult(
            success=True,
            message=f"Path exists ({len(result)} node(s))",
            file_candidates=result.file_candidates,
        )

    async def all_simple_paths(
        self,
        source: str,
        target: str,
        *,
        cutoff: int | None = None,
        session: AsyncSession | None = None,
    ) -> list[list[str]]:
        await self._ensure_fresh(session)
        if source not in self._nodes:
            msg = f"Node not found: {source!r}"
            raise KeyError(msg)
        if target not in self._nodes:
            msg = f"Node not found: {target!r}"
            raise KeyError(msg)
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(
            self._all_simple_paths_impl, nodes, edges, source, target, cutoff
        )

    @staticmethod
    def _all_simple_paths_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
        source: str,
        target: str,
        cutoff: int | None,
    ) -> list[list[str]]:
        graph, path_to_idx, idx_to_path = RustworkxGraph._build_graph_from(nodes, edges)
        src_idx = path_to_idx[source]
        tgt_idx = path_to_idx[target]
        raw = rustworkx.digraph_all_simple_paths(graph, src_idx, tgt_idx, cutoff=cutoff or 0)
        return [[idx_to_path[i] for i in path] for path in raw]

    async def topological_sort(self, *, session: AsyncSession | None = None) -> list[str]:
        await self._ensure_fresh(session)
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(self._topological_sort_impl, nodes, edges)

    @staticmethod
    def _topological_sort_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
    ) -> list[str]:
        graph, _, idx_to_path = RustworkxGraph._build_graph_from(nodes, edges)
        try:
            indices = rustworkx.topological_sort(graph)
        except rustworkx.DAGHasCycle:
            msg = "Graph contains cycles"
            raise ValueError(msg) from None
        return [idx_to_path[i] for i in indices if i in idx_to_path]

    async def shortest_path_length(
        self, source: str, target: str, *, session: AsyncSession | None = None
    ) -> float | None:
        await self._ensure_fresh(session)
        if source not in self._nodes:
            msg = f"Node not found: {source!r}"
            raise KeyError(msg)
        if target not in self._nodes:
            msg = f"Node not found: {target!r}"
            raise KeyError(msg)
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(
            self._shortest_path_length_impl, nodes, edges, source, target
        )

    @staticmethod
    def _shortest_path_length_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
        source: str,
        target: str,
    ) -> float | None:
        graph, path_to_idx, _ = RustworkxGraph._build_graph_from(nodes, edges)
        src_idx = path_to_idx[source]
        tgt_idx = path_to_idx[target]
        lengths = rustworkx.dijkstra_shortest_path_lengths(
            graph, src_idx, lambda _e: 1.0, goal=tgt_idx
        )
        result = dict(lengths)
        return result.get(tgt_idx)

    # --- Meeting subgraph (composite heavy) ---

    async def meeting_subgraph(
        self,
        start_paths: list[str],
        *,
        max_size: int = 50,
        session: AsyncSession | None = None,
    ) -> MeetingSubgraphResult:
        """Find the subgraph connecting *start_paths* via shortest paths."""
        await self._ensure_fresh(session)
        valid_starts = [p for p in start_paths if p in self._nodes]
        if len(valid_starts) <= 1:
            sub = await self.subgraph(valid_starts, session=session)
            return MeetingSubgraphResult(
                success=sub.success,
                message=sub.message,
                file_candidates=sub.file_candidates,
                connection_candidates=sub.connection_candidates,
            )
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(
            self._meeting_subgraph_impl, nodes, edges, valid_starts, max_size
        )

    @staticmethod
    def _meeting_subgraph_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
        valid_starts: list[str],
        max_size: int,
    ) -> MeetingSubgraphResult:
        # Collect all nodes on pairwise shortest paths
        all_nodes: set[str] = set(valid_starts)
        found_connection = False
        for i, src in enumerate(valid_starts):
            for tgt in valid_starts[i + 1 :]:
                path_fwd = RustworkxGraph._path_between_impl(nodes, edges, src, tgt)
                if path_fwd:
                    found_connection = True
                    for p in path_fwd.paths:
                        all_nodes.add(p)
                path_rev = RustworkxGraph._path_between_impl(nodes, edges, tgt, src)
                if path_rev:
                    found_connection = True
                    for p in path_rev.paths:
                        all_nodes.add(p)

        if not found_connection:
            common = RustworkxGraph._common_reachable_impl(nodes, edges, valid_starts, "forward")
            for node in list(common)[:5]:
                all_nodes.add(node)

        # Score with personalized PageRank
        pers = dict.fromkeys(valid_starts, 1.0)
        pr_result = RustworkxGraph._pagerank_impl(nodes, edges, None, 0.85, pers, 100, 1e-6)
        scores = {c.path: c.evidence[0].score for c in pr_result.file_candidates}

        # Prune to max_size
        start_set = set(valid_starts)
        node_list = sorted(all_nodes)
        while len(node_list) > max_size:
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

        # Build subgraph inline
        valid = {p for p in node_list if p in nodes}
        edge_list: list[tuple[str, str, dict[str, Any]]] = []
        for s, t in edges:
            if s in valid and t in valid:
                edge_list.append((s, t, {"type": "", "weight": 1.0}))
        nodes_sorted = sorted(valid)
        sub_fcs = [
            FileCandidate(path=n, evidence=[GraphEvidence(operation="subgraph")])
            for n in nodes_sorted
        ]
        sub_ccs = [
            ConnectionCandidate(
                source_path=s,
                target_path=t,
                connection_type=data.get("type", ""),
                weight=data.get("weight", 1.0),
                evidence=[GraphEvidence(operation="subgraph")],
            )
            for s, t, data in edge_list
        ]
        sub_message = f"{len(nodes_sorted)} node(s), {len(edge_list)} edge(s)"

        # Enrich file_candidates with PageRank scores
        enriched_fcs = [
            FileCandidate(
                path=c.path,
                evidence=[
                    GraphEvidence(
                        operation="min_meeting_subgraph",
                        algorithm="min_meeting_subgraph",
                        score=scores.get(c.path, 0.0),
                    )
                ],
            )
            for c in sub_fcs
        ]
        return MeetingSubgraphResult(
            success=True,
            message=sub_message,
            file_candidates=enriched_fcs,
            connection_candidates=sub_ccs,
        )

    # --- Common reachable ---

    async def common_reachable(
        self,
        paths: list[str],
        *,
        direction: str = "forward",
        session: AsyncSession | None = None,
    ) -> set[str]:
        """Intersection of descendants (forward) or ancestors (reverse)."""
        await self._ensure_fresh(session)
        valid = [p for p in paths if p in self._nodes]
        if not valid:
            return set()
        nodes, edges = self._snapshot()
        return await asyncio.to_thread(self._common_reachable_impl, nodes, edges, valid, direction)

    @staticmethod
    def _common_reachable_impl(
        nodes: frozenset[str],
        edges: frozenset[tuple[str, str]],
        valid: list[str],
        direction: str,
    ) -> set[str]:
        graph, path_to_idx, idx_to_path = RustworkxGraph._build_graph_from(nodes, edges)
        if direction == "forward":
            sets = [
                {
                    idx_to_path[i]
                    for i in rustworkx.descendants(graph, path_to_idx[p])
                    if i in idx_to_path
                }
                for p in valid
            ]
        else:
            sets = [
                {
                    idx_to_path[i]
                    for i in rustworkx.ancestors(graph, path_to_idx[p])
                    if i in idx_to_path
                }
                for p in valid
            ]
        result = sets[0]
        for s in sets[1:]:
            result = result & s
        return result

    # ------------------------------------------------------------------
    # Multi-source BFS helper (static for thread safety)
    # ------------------------------------------------------------------

    @staticmethod
    def _multisource_bfs_static(
        graph: rustworkx.PyDiGraph,
        seed_indices: list[int],
    ) -> set[int]:
        """Multi-source BFS via neighbors_undirected() + Union-Find."""
        if not graph.node_indices():
            return set(seed_indices)
        arr_size = max(graph.node_indices()) + 1
        origin = [-1] * arr_size
        pred = [-1] * arr_size

        uf_parent = list(range(arr_size))
        uf_rank = [0] * arr_size
        remaining = len(seed_indices)

        def uf_find(x: int) -> int:
            while uf_parent[x] != x:
                uf_parent[x] = uf_parent[uf_parent[x]]
                x = uf_parent[x]
            return x

        def uf_union(a: int, b: int) -> bool:
            nonlocal remaining
            ra, rb = uf_find(a), uf_find(b)
            if ra == rb:
                return False
            if uf_rank[ra] < uf_rank[rb]:
                ra, rb = rb, ra
            uf_parent[rb] = ra
            if uf_rank[ra] == uf_rank[rb]:
                uf_rank[ra] += 1
            remaining -= 1
            return True

        queue: deque[int] = deque(seed_indices)
        for s in seed_indices:
            origin[s] = s
            pred[s] = s

        bridges: list[tuple[int, int]] = []
        while queue and remaining > 1:
            node = queue.popleft()
            node_origin = origin[node]
            for neighbor in graph.neighbors_undirected(node):
                if origin[neighbor] == -1:
                    origin[neighbor] = node_origin
                    pred[neighbor] = node
                    queue.append(neighbor)
                elif uf_find(origin[neighbor]) != uf_find(node_origin):
                    bridges.append((node, neighbor))
                    uf_union(origin[neighbor], node_origin)

        keep = set(seed_indices)
        for a, b in bridges:
            for start in (a, b):
                cur = start
                while cur != pred[cur]:
                    keep.add(cur)
                    cur = pred[cur]
                keep.add(cur)
        return keep

    # Keep instance method delegating to static for connecting_subgraph
    def _multisource_bfs(
        self,
        graph: rustworkx.PyDiGraph,
        seed_indices: list[int],
    ) -> set[int]:
        return self._multisource_bfs_static(graph, seed_indices)

    def _graph_from_candidates(self, candidates: FileSearchResult) -> RustworkxGraph:
        """Build a RustworkxGraph from explicit file_candidates + connection_candidates."""
        sub = RustworkxGraph()
        sub._nodes = {c.path for c in candidates.file_candidates}
        sub._edges = {(cc.source_path, cc.target_path) for cc in candidates.connection_candidates}
        return sub

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def from_sql(
        self,
        session: AsyncSession,
        file_model: type | None = None,
        *,
        path_prefix: str = "",
    ) -> None:
        """Load graph state from the database, replacing in-memory state.

        Build-then-swap: new state is assembled in local variables and assigned
        atomically at the end, so concurrent readers never see an empty graph.
        """
        from sqlalchemy import select

        from grover.models.connection import FileConnection

        if file_model is None:
            from grover.models.file import File

            file_model = File

        def _prefix(p: str) -> str:
            if not path_prefix:
                return p
            if p == "/":
                return path_prefix
            return path_prefix + p

        # Build new state in local variables (no mutation of self yet)
        new_nodes: set[str] = set()
        new_edges: set[tuple[str, str]] = set()

        # Load non-deleted files as nodes
        result = await session.execute(
            select(file_model).where(file_model.deleted_at.is_(None))  # type: ignore[union-attr]
        )
        for file_row in result.scalars().all():
            raw_path: str = file_row.path  # type: ignore[unresolved-attribute]
            new_nodes.add(_prefix(raw_path))

        # Load all edges (topology only)
        result = await session.execute(select(FileConnection))
        for edge_row in result.scalars().all():
            src = _prefix(edge_row.source_path)
            tgt = _prefix(edge_row.target_path)
            new_nodes.add(src)
            new_nodes.add(tgt)
            new_edges.add((src, tgt))

        # Atomic swap
        self._nodes = new_nodes
        self._edges = new_edges
        self._loaded_at = time.monotonic()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_node(self, path: str) -> None:
        """Raise ``KeyError`` if *path* is not in the graph."""
        if path not in self._nodes:
            msg = f"Node not found: {path!r}"
            raise KeyError(msg)

    def _undirected_neighbors(self, path: str) -> set[str]:
        if path not in self._nodes:
            msg = f"Node not found: {path!r}"
            raise KeyError(msg)
        neighbors: set[str] = set()
        for s, t in self._edges:
            if s == path:
                neighbors.add(t)
            elif t == path:
                neighbors.add(s)
        return neighbors
