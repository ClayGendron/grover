"""RustworkxGraph — rustworkx-backed graph store implementing GraphProvider protocol."""

from __future__ import annotations

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
    A ``rustworkx.PyDiGraph`` is built on demand via ``_build_graph()``
    and cached until the next mutation.

    Implements the ``GraphProvider`` protocol.
    """

    def __init__(self) -> None:
        self._nodes: set[str] = set()
        self._edges: set[tuple[str, str]] = set()
        self._cached: tuple[rustworkx.PyDiGraph, dict[str, int], dict[int, str]] | None = None

    # ------------------------------------------------------------------
    # Graph construction (lazy, cached)
    # ------------------------------------------------------------------

    def _build_graph(self) -> tuple[rustworkx.PyDiGraph, dict[str, int], dict[int, str]]:
        """Build a PyDiGraph from _nodes/_edges. Cached until mutation."""
        if self._cached is not None:
            return self._cached
        graph: rustworkx.PyDiGraph = rustworkx.PyDiGraph()
        path_to_idx: dict[str, int] = {}
        idx_to_path: dict[int, str] = {}
        for path in self._nodes:
            idx = graph.add_node(path)
            path_to_idx[path] = idx
            idx_to_path[idx] = path
        for source, target in self._edges:
            if source in path_to_idx and target in path_to_idx:
                graph.add_edge(path_to_idx[source], path_to_idx[target], None)
        self._cached = (graph, path_to_idx, idx_to_path)
        return self._cached

    def _invalidate(self) -> None:
        """Clear the cached PyDiGraph after a mutation."""
        self._cached = None

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, path: str, **attrs: object) -> None:
        """Add a node. Extra *attrs* are accepted for protocol compat but not stored."""
        if path not in self._nodes:
            self._nodes.add(path)
            self._invalidate()

    def remove_node(self, path: str) -> None:
        """Remove a node and all incident edges. Raises ``KeyError`` if missing."""
        if path not in self._nodes:
            msg = f"Node not found: {path!r}"
            raise KeyError(msg)
        self._nodes.discard(path)
        self._edges = {(s, t) for s, t in self._edges if s != path and t != path}
        self._invalidate()

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
        **attrs: object,
    ) -> None:
        """Add a directed edge. Auto-creates missing endpoint nodes."""
        changed = False
        if source not in self._nodes:
            self._nodes.add(source)
            changed = True
        if target not in self._nodes:
            self._nodes.add(target)
            changed = True
        if (source, target) not in self._edges:
            self._edges.add((source, target))
            changed = True
        if changed:
            self._invalidate()

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
        self._invalidate()

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
        g, _, _ = self._build_graph()
        return g

    # ------------------------------------------------------------------
    # Query methods — return typed FileSearchResult subclasses
    # ------------------------------------------------------------------

    def predecessors(self, path: str) -> PredecessorsResult:
        """Nodes with edges pointing *to* this node."""
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

    def successors(self, path: str) -> SuccessorsResult:
        """Nodes this node points *to*."""
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

    def path_between(self, source: str, target: str) -> ShortestPathResult:
        """Shortest path (Dijkstra) from *source* to *target*."""
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
        graph, path_to_idx, idx_to_path = self._build_graph()
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

    def contains(self, path: str) -> list[Ref]:
        """Successors as Refs (internal use — not part of typed-result API)."""
        self._require_node(path)
        return [Ref(path=t) for s, t in self._edges if s == path]

    def by_parent(self, parent_path: str) -> list[Ref]:
        """Not supported with minimal storage — returns empty list."""
        return []

    def remove_file_subgraph(self, path: str) -> list[str]:
        """Remove a node and all its successors connected by any edge."""
        if path not in self._nodes:
            msg = f"Node not found: {path!r}"
            raise KeyError(msg)
        # Collect direct successors as children
        children = {t for s, t in self._edges if s == path}
        removed = [path, *sorted(children)]
        for p in removed:
            if p in self._nodes:
                self._nodes.discard(p)
                self._edges = {(s, t) for s, t in self._edges if s != p and t != p}
        self._invalidate()
        return removed

    # ------------------------------------------------------------------
    # Graph-level
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
    # Centrality algorithms — return typed FileSearchResult subclasses
    # ------------------------------------------------------------------

    def _scores_to_candidates(
        self, scores: dict[str, float], operation: str, algorithm: str
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

    def pagerank(
        self,
        candidates: FileSearchResult | None = None,
        *,
        alpha: float = 0.85,
        personalization: dict[str, float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> PageRankResult:
        """PageRank centrality scores."""
        graph, path_to_idx, idx_to_path = self._resolve_graph(candidates)
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
        fcs = self._scores_to_candidates(raw, "pagerank", "pagerank")
        return PageRankResult(
            success=True,
            message=f"{len(fcs)} node(s)",
            file_candidates=fcs,
        )

    def betweenness_centrality(
        self,
        candidates: FileSearchResult | None = None,
        *,
        normalized: bool = True,
    ) -> BetweennessResult:
        """Betweenness centrality scores."""
        graph, _, idx_to_path = self._resolve_graph(candidates)
        if graph.num_nodes() == 0:
            return BetweennessResult(success=True, message="0 node(s)")
        scores = rustworkx.digraph_betweenness_centrality(graph, normalized=normalized)
        raw = {idx_to_path[idx]: score for idx, score in scores.items() if idx in idx_to_path}
        fcs = self._scores_to_candidates(raw, "betweenness_centrality", "betweenness_centrality")
        return BetweennessResult(
            success=True,
            message=f"{len(fcs)} node(s)",
            file_candidates=fcs,
        )

    def closeness_centrality(
        self,
        candidates: FileSearchResult | None = None,
    ) -> ClosenessResult:
        """Closeness centrality scores."""
        graph, _, idx_to_path = self._resolve_graph(candidates)
        if graph.num_nodes() == 0:
            return ClosenessResult(success=True, message="0 node(s)")
        scores = rustworkx.closeness_centrality(graph)
        raw = {idx_to_path[idx]: score for idx, score in scores.items() if idx in idx_to_path}
        fcs = self._scores_to_candidates(raw, "closeness_centrality", "closeness_centrality")
        return ClosenessResult(
            success=True,
            message=f"{len(fcs)} node(s)",
            file_candidates=fcs,
        )

    def harmonic_centrality(
        self,
        candidates: FileSearchResult | None = None,
    ) -> HarmonicResult:
        """Harmonic centrality scores."""
        graph, path_to_idx, _idx_to_path = self._resolve_graph(candidates)
        raw: dict[str, float] = {}
        for path, idx in path_to_idx.items():
            lengths = rustworkx.dijkstra_shortest_path_lengths(graph, idx, lambda _e: 1.0)
            score = sum(1.0 / d for d in dict(lengths).values() if d > 0)
            raw[path] = score
        fcs = self._scores_to_candidates(raw, "harmonic_centrality", "harmonic_centrality")
        return HarmonicResult(
            success=True,
            message=f"{len(fcs)} node(s)",
            file_candidates=fcs,
        )

    def hits(
        self,
        candidates: FileSearchResult | None = None,
        *,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> HitsResult:
        """HITS hub and authority scores."""
        graph, path_to_idx, idx_to_path = self._resolve_graph(candidates)
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

    def katz_centrality(
        self,
        candidates: FileSearchResult | None = None,
        *,
        alpha: float = 0.1,
        beta: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> KatzResult:
        """Katz centrality scores."""
        graph, _, idx_to_path = self._resolve_graph(candidates)
        if graph.num_nodes() == 0:
            return KatzResult(success=True, message="0 node(s)")
        scores = rustworkx.katz_centrality(
            graph, alpha=alpha, beta=beta, max_iter=max_iter, tol=tol
        )
        raw = {idx_to_path[idx]: score for idx, score in scores.items() if idx in idx_to_path}
        fcs = self._scores_to_candidates(raw, "katz_centrality", "katz_centrality")
        return KatzResult(
            success=True,
            message=f"{len(fcs)} node(s)",
            file_candidates=fcs,
        )

    def degree_centrality(
        self,
        candidates: FileSearchResult | None = None,
    ) -> DegreeResult:
        """Degree centrality (in + out) scores."""
        graph, _, idx_to_path = self._resolve_graph(candidates)
        if graph.num_nodes() == 0:
            return DegreeResult(success=True, message="0 node(s)")
        scores = rustworkx.digraph_degree_centrality(graph)
        raw = {idx_to_path[idx]: score for idx, score in scores.items() if idx in idx_to_path}
        fcs = self._scores_to_candidates(raw, "degree_centrality", "degree_centrality")
        return DegreeResult(
            success=True,
            message=f"{len(fcs)} node(s)",
            file_candidates=fcs,
        )

    def in_degree_centrality(
        self,
        candidates: FileSearchResult | None = None,
    ) -> DegreeResult:
        """In-degree centrality scores."""
        graph, _, idx_to_path = self._resolve_graph(candidates)
        if graph.num_nodes() == 0:
            return DegreeResult(success=True, message="0 node(s)")
        scores = rustworkx.in_degree_centrality(graph)
        raw = {idx_to_path[idx]: score for idx, score in scores.items() if idx in idx_to_path}
        fcs = self._scores_to_candidates(raw, "in_degree_centrality", "in_degree_centrality")
        return DegreeResult(
            success=True,
            message=f"{len(fcs)} node(s)",
            file_candidates=fcs,
        )

    def out_degree_centrality(
        self,
        candidates: FileSearchResult | None = None,
    ) -> DegreeResult:
        """Out-degree centrality scores."""
        graph, _, idx_to_path = self._resolve_graph(candidates)
        if graph.num_nodes() == 0:
            return DegreeResult(success=True, message="0 node(s)")
        scores = rustworkx.out_degree_centrality(graph)
        raw = {idx_to_path[idx]: score for idx, score in scores.items() if idx in idx_to_path}
        fcs = self._scores_to_candidates(raw, "out_degree_centrality", "out_degree_centrality")
        return DegreeResult(
            success=True,
            message=f"{len(fcs)} node(s)",
            file_candidates=fcs,
        )

    # ------------------------------------------------------------------
    # Connectivity algorithms
    # ------------------------------------------------------------------

    def weakly_connected_components(self) -> list[set[str]]:
        graph, _, idx_to_path = self._build_graph()
        components = rustworkx.weakly_connected_components(graph)
        return [{idx_to_path[idx] for idx in comp if idx in idx_to_path} for comp in components]

    def strongly_connected_components(self) -> list[set[str]]:
        graph, _, idx_to_path = self._build_graph()
        components = rustworkx.strongly_connected_components(graph)
        return [{idx_to_path[idx] for idx in comp if idx in idx_to_path} for comp in components]

    def is_weakly_connected(self) -> bool:
        graph, _, _ = self._build_graph()
        try:
            return rustworkx.is_weakly_connected(graph)
        except rustworkx.NullGraph:
            return True

    # ------------------------------------------------------------------
    # Traversal algorithms — return typed results
    # ------------------------------------------------------------------

    def ancestors(self, path: str) -> AncestorsResult:
        self._require_node(path)
        graph, path_to_idx, idx_to_path = self._build_graph()
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

    def descendants(self, path: str) -> DescendantsResult:
        self._require_node(path)
        graph, path_to_idx, idx_to_path = self._build_graph()
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

    def all_simple_paths(
        self,
        source: str,
        target: str,
        *,
        cutoff: int | None = None,
    ) -> list[list[str]]:
        if source not in self._nodes:
            msg = f"Node not found: {source!r}"
            raise KeyError(msg)
        if target not in self._nodes:
            msg = f"Node not found: {target!r}"
            raise KeyError(msg)
        graph, path_to_idx, idx_to_path = self._build_graph()
        src_idx = path_to_idx[source]
        tgt_idx = path_to_idx[target]
        raw = rustworkx.digraph_all_simple_paths(graph, src_idx, tgt_idx, cutoff=cutoff or 0)
        return [[idx_to_path[i] for i in path] for path in raw]

    def topological_sort(self) -> list[str]:
        graph, _, idx_to_path = self._build_graph()
        try:
            indices = rustworkx.topological_sort(graph)
        except rustworkx.DAGHasCycle:
            msg = "Graph contains cycles"
            raise ValueError(msg) from None
        return [idx_to_path[i] for i in indices if i in idx_to_path]

    def shortest_path_length(self, source: str, target: str) -> float | None:
        if source not in self._nodes:
            msg = f"Node not found: {source!r}"
            raise KeyError(msg)
        if target not in self._nodes:
            msg = f"Node not found: {target!r}"
            raise KeyError(msg)
        graph, path_to_idx, _ = self._build_graph()
        src_idx = path_to_idx[source]
        tgt_idx = path_to_idx[target]
        lengths = rustworkx.dijkstra_shortest_path_lengths(
            graph, src_idx, lambda _e: 1.0, goal=tgt_idx
        )
        result = dict(lengths)
        return result.get(tgt_idx)

    def has_path(self, source: str, target: str) -> HasPathResult:
        result = self.path_between(source, target)
        if not result:
            return HasPathResult(success=True, message="No path exists")
        return HasPathResult(
            success=True,
            message=f"Path exists ({len(result)} node(s))",
            file_candidates=result.file_candidates,
        )

    # ------------------------------------------------------------------
    # Subgraph extraction — return typed results
    # ------------------------------------------------------------------

    def subgraph(self, paths: list[str]) -> SubgraphSearchResult:
        """Extract the induced subgraph for the given *paths*."""
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

    def neighborhood(
        self,
        path: str,
        *,
        max_depth: int = 2,
        direction: str = "both",
        edge_types: list[str] | None = None,
    ) -> EgoGraphResult:
        """BFS neighborhood around *path* up to *max_depth* hops."""
        self._require_node(path)
        # Build adjacency for BFS
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

        sub = self.subgraph(sorted(visited))
        return EgoGraphResult(
            success=sub.success,
            message=sub.message,
            file_candidates=sub.file_candidates,
            connection_candidates=sub.connection_candidates,
        )

    def meeting_subgraph(
        self,
        start_paths: list[str],
        *,
        max_size: int = 50,
    ) -> MeetingSubgraphResult:
        """Find the subgraph connecting *start_paths* via shortest paths."""
        valid_starts = [p for p in start_paths if p in self._nodes]
        if len(valid_starts) <= 1:
            sub = self.subgraph(valid_starts)
            return MeetingSubgraphResult(
                success=sub.success,
                message=sub.message,
                file_candidates=sub.file_candidates,
                connection_candidates=sub.connection_candidates,
            )

        # Collect all nodes on pairwise shortest paths
        all_nodes: set[str] = set(valid_starts)
        found_connection = False
        for i, src in enumerate(valid_starts):
            for tgt in valid_starts[i + 1 :]:
                path_fwd = self.path_between(src, tgt)
                if path_fwd:
                    found_connection = True
                    for p in path_fwd.paths:
                        all_nodes.add(p)
                path_rev = self.path_between(tgt, src)
                if path_rev:
                    found_connection = True
                    for p in path_rev.paths:
                        all_nodes.add(p)

        if not found_connection:
            common = self.common_reachable(valid_starts, direction="forward")
            for node in list(common)[:5]:
                all_nodes.add(node)

        # Score with personalized PageRank
        pers = dict.fromkeys(valid_starts, 1.0)
        pr_result = self.pagerank(personalization=pers)
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

        sub = self.subgraph(node_list)
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
            for c in sub.file_candidates
        ]
        return MeetingSubgraphResult(
            success=True,
            message=sub.message,
            file_candidates=enriched_fcs,
            connection_candidates=sub.connection_candidates,
        )

    def common_reachable(
        self,
        paths: list[str],
        *,
        direction: str = "forward",
    ) -> set[str]:
        """Intersection of descendants (forward) or ancestors (reverse)."""
        valid = [p for p in paths if p in self._nodes]
        if not valid:
            return set()
        if direction == "forward":
            sets = [set(self.descendants(p).paths) for p in valid]
        else:
            sets = [set(self.ancestors(p).paths) for p in valid]
        result = sets[0]
        for s in sets[1:]:
            result = result & s
        return result

    def common_neighbors(self, path1: str, path2: str) -> CommonNeighborsResult:
        """Intersection of undirected neighbors of both nodes."""
        neighbors = sorted(self._undirected_neighbors(path1) & self._undirected_neighbors(path2))
        return CommonNeighborsResult(
            success=True,
            message=f"{len(neighbors)} common neighbor(s)",
            file_candidates=[
                FileCandidate(path=p, evidence=[GraphEvidence(operation="common_neighbors")])
                for p in neighbors
            ],
        )

    # ------------------------------------------------------------------
    # Connecting subgraph — multi-source BFS + Union-Find
    # ------------------------------------------------------------------

    def connecting_subgraph(self, paths: list[str]) -> RustworkxGraph:
        """Return a new RustworkxGraph containing all nodes needed to connect *paths*.

        Uses multi-source BFS with Union-Find for O(V+E) performance.
        """
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

    def _multisource_bfs(
        self,
        graph: rustworkx.PyDiGraph,
        seed_indices: list[int],
    ) -> set[int]:
        """Multi-source BFS via neighbors_undirected() + Union-Find.

        Returns set of node indices connecting all seeds.
        """
        if not graph.node_indices():
            return set(seed_indices)
        arr_size = max(graph.node_indices()) + 1
        origin = [-1] * arr_size
        pred = [-1] * arr_size

        # Inline Union-Find
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

        # Trace predecessor chains from bridge endpoints back to seeds
        keep = set(seed_indices)
        for a, b in bridges:
            for start in (a, b):
                cur = start
                while cur != pred[cur]:
                    keep.add(cur)
                    cur = pred[cur]
                keep.add(cur)
        return keep

    def _graph_from_candidates(self, candidates: FileSearchResult) -> RustworkxGraph:
        """Build a RustworkxGraph from explicit file_candidates + connection_candidates."""
        sub = RustworkxGraph()
        sub._nodes = {c.path for c in candidates.file_candidates}
        sub._edges = {(cc.source_path, cc.target_path) for cc in candidates.connection_candidates}
        return sub

    # ------------------------------------------------------------------
    # Helper: resolve graph for centrality (subgraph or full)
    # ------------------------------------------------------------------

    def _resolve_graph(
        self,
        candidates: FileSearchResult | None,
    ) -> tuple[rustworkx.PyDiGraph, dict[str, int], dict[int, str]]:
        """Return the (graph, path_to_idx, idx_to_path) to run centrality on.

        Three modes:
        1. candidates is None → full graph
        2. candidates has connection_candidates → build from explicit edges
        3. candidates has file_candidates only → connecting_subgraph
        """
        if candidates is None:
            return self._build_graph()
        if candidates.connection_candidates:
            sub = self._graph_from_candidates(candidates)
            return sub._build_graph()
        paths = [c.path for c in candidates.file_candidates]
        if not paths:
            return self._build_graph()
        sub = self.connecting_subgraph(paths)
        return sub._build_graph()

    # ------------------------------------------------------------------
    # Node similarity
    # ------------------------------------------------------------------

    def node_similarity(
        self,
        path1: str,
        path2: str,
        *,
        method: str = "jaccard",
    ) -> float:
        n1 = self._undirected_neighbors(path1)
        n2 = self._undirected_neighbors(path2)
        union = n1 | n2
        if not union:
            return 0.0
        return len(n1 & n2) / len(union)

    def similar_nodes(
        self,
        path: str,
        *,
        method: str = "jaccard",
        k: int = 10,
    ) -> list[tuple[str, float]]:
        scores: list[tuple[str, float]] = []
        for other in self._nodes:
            if other == path:
                continue
            s = self.node_similarity(path, other, method=method)
            scores.append((other, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

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

        Loads non-deleted file rows as nodes and all ``FileConnection``
        rows as edges (topology only — no metadata stored).
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

        # Reset
        self._nodes = set()
        self._edges = set()
        self._cached = None

        # Load non-deleted files as nodes
        result = await session.execute(
            select(file_model).where(file_model.deleted_at.is_(None))  # type: ignore[union-attr]
        )
        for file_row in result.scalars().all():
            raw_path: str = file_row.path  # type: ignore[unresolved-attribute]
            self._nodes.add(_prefix(raw_path))

        # Load all edges (topology only)
        result = await session.execute(select(FileConnection))
        for edge_row in result.scalars().all():
            src = _prefix(edge_row.source_path)
            tgt = _prefix(edge_row.target_path)
            # Auto-create nodes for dangling endpoints
            self._nodes.add(src)
            self._nodes.add(tgt)
            self._edges.add((src, tgt))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_node(self, path: str) -> None:
        """Raise ``KeyError`` if *path* is not in the graph."""
        if path not in self._nodes:
            msg = f"Node not found: {path!r}"
            raise KeyError(msg)
