"""RustworkxGraph — rustworkx-backed graph implementing GraphProvider.

Stores topology as ``_nodes`` (set of paths) and adjacency dicts
``_out`` (source → targets) / ``_in`` (target → sources).
No tuple-per-edge allocation — O(degree) lookups for predecessors
and successors instead of O(|E|) scans.

Query/algorithm methods are ``async def``:
- Light reads run inline (no thread overhead).
- Heavy algorithms use ``asyncio.to_thread`` with a snapshot for concurrency.

Mutations stay synchronous (trivial set operations, called from background tasks).
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import rustworkx
from sqlmodel import select

from grover.paths import connection_path, decompose_connection
from grover.results import Candidate, Detail, GroverResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models import GroverObjectBase


# ---------------------------------------------------------------------------
# Union-Find — used by meeting_subgraph
# ---------------------------------------------------------------------------


class UnionFind:
    """Path-compressed union-find with rank balancing."""

    __slots__ = ("components", "parent", "rank")

    def __init__(self, elements: list[str]) -> None:
        self.parent = {e: e for e in elements}
        self.rank = dict.fromkeys(elements, 0)
        self.components = len(self.parent)

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.components -= 1
        return True


# ---------------------------------------------------------------------------
# RustworkxGraph
# ---------------------------------------------------------------------------


class RustworkxGraph:
    """Directed knowledge graph over file paths.

    Implements the ``GraphProvider`` protocol.
    """

    DEFAULT_TTL: float = 3600  # 1 hour

    def __init__(self, model: type[GroverObjectBase], *, ttl: float | None = None) -> None:
        self._model = model
        self._ttl = ttl if ttl is not None else self.DEFAULT_TTL
        self._nodes: set[str] = set()
        self._out: dict[str, set[str]] = {}  # source → targets
        self._in: dict[str, set[str]] = {}  # target → sources
        self._edge_types: dict[tuple[str, str], str] = {}  # (source, target) → type
        self._loaded_at: float | None = None

    def __repr__(self) -> str:
        edge_count = sum(len(ts) for ts in self._out.values())
        return f"RustworkxGraph(nodes={len(self._nodes)}, edges={edge_count})"

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    async def add_node(self, path: str, *, session: AsyncSession) -> None:
        """Add a node. Idempotent."""
        await self.ensure_fresh(session)
        self._nodes.add(path)

    async def remove_node(self, path: str, *, session: AsyncSession) -> None:
        """Remove a node and all incident edges. Raises ``KeyError`` if missing."""
        await self.ensure_fresh(session)
        if path not in self._nodes:
            msg = f"Node not found: {path!r}"
            raise KeyError(msg)
        self._nodes.discard(path)
        # Remove outgoing edges
        for t in self._out.pop(path, set()):
            self._edge_types.pop((path, t), None)
            in_set = self._in.get(t)
            if in_set is not None:
                in_set.discard(path)
                if not in_set:
                    del self._in[t]
        # Remove incoming edges
        for s in self._in.pop(path, set()):
            self._edge_types.pop((s, path), None)
            out_set = self._out.get(s)
            if out_set is not None:
                out_set.discard(path)
                if not out_set:
                    del self._out[s]

    async def has_node(self, path: str, *, session: AsyncSession) -> bool:
        await self.ensure_fresh(session)
        return path in self._nodes

    async def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        *,
        weight: float = 1.0,
        session: AsyncSession,
    ) -> None:
        """Add a directed edge. Auto-creates missing endpoint nodes."""
        await self.ensure_fresh(session)
        self._nodes.add(source)
        self._nodes.add(target)
        self._out.setdefault(source, set()).add(target)
        self._in.setdefault(target, set()).add(source)
        self._edge_types[(source, target)] = edge_type

    async def remove_edge(self, source: str, target: str, *, session: AsyncSession) -> None:
        """Remove the edge between *source* and *target*. Raises ``KeyError``."""
        await self.ensure_fresh(session)
        if source not in self._nodes:
            msg = f"Node not found: {source!r}"
            raise KeyError(msg)
        if target not in self._nodes:
            msg = f"Node not found: {target!r}"
            raise KeyError(msg)
        if target not in self._out.get(source, set()):
            msg = f"No edge from {source!r} to {target!r}"
            raise KeyError(msg)
        out_set = self._out.get(source)
        if out_set is not None:
            out_set.discard(target)
            if not out_set:
                del self._out[source]
        in_set = self._in.get(target)
        if in_set is not None:
            in_set.discard(source)
            if not in_set:
                del self._in[target]
        self._edge_types.pop((source, target), None)

    async def has_edge(self, source: str, target: str, *, session: AsyncSession) -> bool:
        await self.ensure_fresh(session)
        return target in self._out.get(source, set())

    @property
    def nodes(self) -> set[str]:
        return self._nodes

    # ------------------------------------------------------------------
    # Snapshot and graph construction helpers
    # ------------------------------------------------------------------

    def _snapshot(self) -> tuple[frozenset[str], dict[str, frozenset[str]]]:
        """Return immutable copies of nodes and edges for thread-safe reads."""
        return (
            frozenset(self._nodes),
            {s: frozenset(ts) for s, ts in self._out.items()},
        )

    @staticmethod
    def _build_graph_from(
        nodes: frozenset[str],
        edges_out: dict[str, frozenset[str]],
    ) -> tuple[rustworkx.PyDiGraph, dict[str, int], dict[int, str]]:
        """Build a PyDiGraph from node set and adjacency dict."""
        graph: rustworkx.PyDiGraph = rustworkx.PyDiGraph()
        path_to_idx: dict[str, int] = {}
        idx_to_path: dict[int, str] = {}
        for path in nodes:
            idx = graph.add_node(path)
            path_to_idx[path] = idx
            idx_to_path[idx] = path
        for source, targets in edges_out.items():
            src_idx = path_to_idx.get(source)
            if src_idx is None:
                continue
            for target in targets:
                tgt_idx = path_to_idx.get(target)
                if tgt_idx is not None:
                    graph.add_edge(src_idx, tgt_idx, None)
        return graph, path_to_idx, idx_to_path

    async def graph(self, *, session: AsyncSession) -> rustworkx.PyDiGraph:
        """Access the underlying rustworkx directed graph."""
        await self.ensure_fresh(session)
        nodes, edges_out = self._snapshot()
        g, _, _ = self._build_graph_from(nodes, edges_out)
        return g

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def ensure_fresh(self, session: AsyncSession) -> None:
        """Load from DB if never loaded or TTL has expired."""
        if self._loaded_at is not None and (time.monotonic() - self._loaded_at) < self._ttl:
            return
        await self._load(session)

    async def _load(self, session: AsyncSession) -> None:
        """Load graph state from GroverObject connection rows.

        Build-then-swap: new state is assembled in local variables and
        assigned atomically so concurrent readers never see an empty graph.
        """
        new_nodes: set[str] = set()
        new_out: dict[str, set[str]] = {}
        new_in: dict[str, set[str]] = {}
        new_edge_types: dict[tuple[str, str], str] = {}

        stmt = select(self._model).where(self._model.kind == "connection")
        result = await session.execute(stmt)
        rows: list[GroverObjectBase] = list(result.scalars().all())
        for obj in rows:
            src = obj.source_path
            tgt = obj.target_path
            parts = decompose_connection(obj.path)
            conn_type = obj.connection_type or (parts.connection_type if parts else "")
            if src and tgt:
                new_nodes.add(src)
                new_nodes.add(tgt)
                new_out.setdefault(src, set()).add(tgt)
                new_in.setdefault(tgt, set()).add(src)
                new_edge_types[(src, tgt)] = conn_type

        # Atomic swap
        self._nodes = new_nodes
        self._out = new_out
        self._in = new_in
        self._edge_types = new_edge_types
        self._loaded_at = time.monotonic()

    # ------------------------------------------------------------------
    # Result construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _relationship_candidates(
        paths_dict: dict[str, list[str]],
        operation: str,
    ) -> list[Candidate]:
        """Build candidates from {path: [related_paths]} mapping."""
        return [
            Candidate(
                path=p,
                details=(
                    Detail(
                        operation=operation,
                        metadata={"paths": sorted(paths_dict[p])},
                    ),
                ),
            )
            for p in sorted(paths_dict)
        ]

    @staticmethod
    def _subgraph_candidates(
        node_set: set[str],
        edges_out: dict[str, frozenset[str]],
        edge_types: dict[tuple[str, str], str],
        operation: str,
    ) -> list[Candidate]:
        """Build node + connection candidates from a subgraph."""
        detail = Detail(operation=operation)

        # Nodes
        candidates: list[Candidate] = [
            Candidate(path=p, details=(detail,)) for p in sorted(node_set)
        ]

        # Edges as connection candidates
        candidates.extend(
            Candidate(
                path=connection_path(s, t, edge_types[(s, t)]),
                weight=1.0,
                details=(detail,),
            )
            for s in node_set
            for t in edges_out.get(s, frozenset())
            if t in node_set
        )

        return candidates

    @staticmethod
    def _score_candidates(
        scores: dict[str, float],
        operation: str,
    ) -> list[Candidate]:
        """Build candidates from {path: score} mapping, sorted by score descending."""
        sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            Candidate(
                path=path,
                details=(Detail(operation=operation, score=score),),
            )
            for path, score in sorted_items
        ]

    @staticmethod
    def _extract_paths(candidates: GroverResult) -> list[str]:
        """Extract path strings from a GroverResult."""
        return [c.path for c in candidates.candidates]

    # ------------------------------------------------------------------
    # Light reads — async inline (no thread overhead)
    # ------------------------------------------------------------------

    async def predecessors(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """One-hop backward: nodes with edges pointing to any candidate."""
        try:
            await self.ensure_fresh(session)
            query_paths = set(self._extract_paths(candidates)) & self._nodes
            predecessor_targets: dict[str, list[str]] = {}

            for t in query_paths:
                for s in self._in.get(t, set()):
                    if s not in query_paths:
                        predecessor_targets.setdefault(s, []).append(t)

            return GroverResult(
                candidates=self._relationship_candidates(predecessor_targets, "predecessors"),
            )

        except Exception as e:
            return GroverResult(success=False, errors=[f"predecessors failed: {e}"])

    async def successors(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """One-hop forward: nodes that any candidate points to."""
        try:
            await self.ensure_fresh(session)
            query_paths = set(self._extract_paths(candidates)) & self._nodes
            successor_sources: dict[str, list[str]] = {}

            for s in query_paths:
                for t in self._out.get(s, set()):
                    if t not in query_paths:
                        successor_sources.setdefault(t, []).append(s)

            return GroverResult(
                candidates=self._relationship_candidates(successor_sources, "successors"),
            )

        except Exception as e:
            return GroverResult(success=False, errors=[f"successors failed: {e}"])

    # ------------------------------------------------------------------
    # Heavy traversal — async via to_thread
    # ------------------------------------------------------------------

    async def ancestors(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """Transitive backward: union of all ancestor sets, excluding candidates."""
        try:
            await self.ensure_fresh(session)
            valid_paths = set(self._extract_paths(candidates)) & self._nodes
            if not valid_paths:
                return GroverResult()
            nodes, edges_out = self._snapshot()
            return await asyncio.to_thread(
                self._ancestors_impl, nodes, edges_out, valid_paths,
            )

        except Exception as e:
            return GroverResult(success=False, errors=[f"ancestors failed: {e}"])

    @staticmethod
    def _ancestors_impl(
        nodes: frozenset[str],
        edges_out: dict[str, frozenset[str]],
        valid_paths: set[str],
    ) -> GroverResult:
        graph, path_to_idx, idx_to_path = RustworkxGraph._build_graph_from(nodes, edges_out)
        result_map: dict[str, list[str]] = {}
        for candidate in valid_paths:
            for i in rustworkx.ancestors(graph, path_to_idx[candidate]):
                p = idx_to_path.get(i)
                if p is not None and p not in valid_paths:
                    result_map.setdefault(p, []).append(candidate)
        return GroverResult(
            candidates=RustworkxGraph._relationship_candidates(result_map, "ancestors"),
        )

    async def descendants(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """Transitive forward: union of all descendant sets, excluding candidates."""
        try:
            await self.ensure_fresh(session)
            valid_paths = set(self._extract_paths(candidates)) & self._nodes
            if not valid_paths:
                return GroverResult()
            nodes, edges_out = self._snapshot()
            return await asyncio.to_thread(
                self._descendants_impl, nodes, edges_out, valid_paths,
            )

        except Exception as e:
            return GroverResult(success=False, errors=[f"descendants failed: {e}"])

    @staticmethod
    def _descendants_impl(
        nodes: frozenset[str],
        edges_out: dict[str, frozenset[str]],
        valid_paths: set[str],
    ) -> GroverResult:
        graph, path_to_idx, idx_to_path = RustworkxGraph._build_graph_from(nodes, edges_out)
        result_map: dict[str, list[str]] = {}
        for candidate in valid_paths:
            for i in rustworkx.descendants(graph, path_to_idx[candidate]):
                p = idx_to_path.get(i)
                if p is not None and p not in valid_paths:
                    result_map.setdefault(p, []).append(candidate)
        return GroverResult(
            candidates=RustworkxGraph._relationship_candidates(result_map, "descendants"),
        )

    async def neighborhood(
        self,
        candidates: GroverResult,
        *,
        depth: int = 2,
        session: AsyncSession,
    ) -> GroverResult:
        """Bounded undirected BFS around candidate nodes."""
        try:
            await self.ensure_fresh(session)
            seed_paths = set(self._extract_paths(candidates)) & self._nodes
            if not seed_paths:
                return GroverResult()

            _, snap_out = self._snapshot()
            snap_in = {t: frozenset(ss) for t, ss in self._in.items()}
            snap_edge_types = self._edge_types.copy()

            visited: set[str] = set(seed_paths)
            frontier: set[str] = set(seed_paths)
            for _ in range(depth):
                next_frontier: set[str] = set()
                for node in frontier:
                    for n in snap_out.get(node, ()):
                        if n not in visited:
                            visited.add(n)
                            next_frontier.add(n)
                    for n in snap_in.get(node, ()):
                        if n not in visited:
                            visited.add(n)
                            next_frontier.add(n)
                frontier = next_frontier
                if not frontier:
                    break

            visited_edges = {s: ts for s, ts in snap_out.items() if s in visited}
            return GroverResult(
                candidates=self._subgraph_candidates(
                    visited, visited_edges, snap_edge_types, "neighborhood",
                ),
            )

        except Exception as e:
            return GroverResult(success=False, errors=[f"neighborhood failed: {e}"])

    # ------------------------------------------------------------------
    # Subgraph algorithms
    # ------------------------------------------------------------------

    async def meeting_subgraph(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """Find minimal subgraph connecting all candidate nodes.

        Uses multi-source BFS with union-find to detect when seed
        wavefronts meet, then leaf-strips non-seed nodes. O(V+E).
        """
        try:
            await self.ensure_fresh(session)
            valid_seeds = [p for p in self._extract_paths(candidates) if p in self._nodes]
            if len(valid_seeds) <= 1:
                detail = Detail(operation="meeting_subgraph")
                return GroverResult(
                    candidates=[Candidate(path=p, details=(detail,)) for p in valid_seeds],
                )
            _, edges_out = self._snapshot()
            edges_in = {t: frozenset(ss) for t, ss in self._in.items()}
            return await asyncio.to_thread(
                self._meeting_subgraph_impl, edges_out, edges_in,
                valid_seeds, self._edge_types.copy(),
            )

        except Exception as e:
            return GroverResult(success=False, errors=[f"meeting_subgraph failed: {e}"])

    @staticmethod
    def _meeting_subgraph_impl(
        edges_out: dict[str, frozenset[str]],
        edges_in: dict[str, frozenset[str]],
        seeds: list[str],
        edge_types: dict[tuple[str, str], str],
    ) -> GroverResult:
        seed_set = set(seeds)

        # Multi-source BFS with union-find
        origin: dict[str, str] = {}
        pred: dict[str, str] = {}
        uf = UnionFind(seeds)

        queue: deque[str] = deque()
        for s in seeds:
            origin[s] = s
            pred[s] = s
            queue.append(s)

        bridges: list[tuple[str, str]] = []

        while queue and uf.components > 1:
            node = queue.popleft()
            node_origin = origin[node]
            for neighbor in edges_out.get(node, ()):
                if neighbor not in origin:
                    origin[neighbor] = node_origin
                    pred[neighbor] = node
                    queue.append(neighbor)
                elif uf.find(origin[neighbor]) != uf.find(node_origin):
                    bridges.append((node, neighbor))
                    uf.union(origin[neighbor], node_origin)
            for neighbor in edges_in.get(node, ()):
                if neighbor not in origin:
                    origin[neighbor] = node_origin
                    pred[neighbor] = node
                    queue.append(neighbor)
                elif uf.find(origin[neighbor]) != uf.find(node_origin):
                    bridges.append((node, neighbor))
                    uf.union(origin[neighbor], node_origin)

        # Trace predecessor chains from bridge endpoints back to seeds
        kept: set[str] = set(seeds)
        for a, b in bridges:
            for start in (a, b):
                node = start
                while node != pred[node]:
                    kept.add(node)
                    node = pred[node]
                kept.add(node)

        # Leaf stripping — remove non-seed leaves iteratively
        kept = RustworkxGraph._strip_leaves(kept, edges_out, edges_in, seed_set)

        return GroverResult(
            candidates=RustworkxGraph._subgraph_candidates(
                kept, edges_out, edge_types, "meeting_subgraph",
            ),
        )

    @staticmethod
    def _strip_leaves(
        kept: set[str],
        edges_out: dict[str, frozenset[str]],
        edges_in: dict[str, frozenset[str]],
        protected: set[str],
    ) -> set[str]:
        """Remove non-protected leaf nodes iteratively. O(n+e)."""
        succs: dict[str, set[str]] = {}
        preds: dict[str, set[str]] = {}
        for s in kept:
            succs[s] = {t for t in edges_out.get(s, ()) if t in kept}
        for t in kept:
            preds[t] = {s for s in edges_in.get(t, ()) if s in kept}

        queue = [
            n for n in kept
            if n not in protected and (not succs.get(n) or not preds.get(n))
        ]
        removed: set[str] = set()
        while queue:
            node = queue.pop()
            if node in removed or node in protected:
                continue
            removed.add(node)
            for succ in succs.get(node, ()):
                preds[succ].discard(node)
                if (
                    succ not in protected
                    and succ not in removed
                    and (not preds[succ] or not succs[succ])
                ):
                    queue.append(succ)
            for pred_node in preds.get(node, ()):
                succs[pred_node].discard(node)
                if (
                    pred_node not in protected
                    and pred_node not in removed
                    and (not succs[pred_node] or not preds[pred_node])
                ):
                    queue.append(pred_node)
        return kept - removed

    async def min_meeting_subgraph(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """Pruned meeting subgraph — drop non-candidate nodes while staying connected.

        Uses ``rustworkx.articulation_points`` to identify nodes whose removal
        would disconnect the graph. Non-seed, non-articulation-point nodes are
        removed one at a time (recomputing articulation points after each
        removal) until only seeds and structurally critical intermediaries
        remain.
        """
        try:
            meeting = await self.meeting_subgraph(candidates, session=session)
            if not meeting.success:
                return meeting

            candidate_paths = set(self._extract_paths(candidates))
            # Extract node set from meeting result — connection paths
            # contain /.connections/, node paths don't.
            node_set = {c.path for c in meeting.candidates if not decompose_connection(c.path)}

            # If meeting subgraph is already minimal, return it
            if len(node_set) <= len(candidate_paths):
                return meeting

            # Build edge topology from stored edge types (authoritative)
            edges_out: dict[str, set[str]] = {}
            for (s, t) in self._edge_types:
                if s in node_set and t in node_set:
                    edges_out.setdefault(s, set()).add(t)

            return await asyncio.to_thread(
                self._min_meeting_impl, node_set, edges_out,
                candidate_paths, self._edge_types.copy(),
            )

        except Exception as e:
            return GroverResult(success=False, errors=[f"min_meeting_subgraph failed: {e}"])

    @staticmethod
    def _min_meeting_impl(
        node_set: set[str],
        edges_out: dict[str, set[str]],
        candidate_paths: set[str],
        edge_types: dict[tuple[str, str], str],
    ) -> GroverResult:
        # Build undirected PyGraph for articulation point detection
        graph = rustworkx.PyGraph()
        path_to_idx: dict[str, int] = {}
        idx_to_path: dict[int, str] = {}
        for p in node_set:
            idx = graph.add_node(p)
            path_to_idx[p] = idx
            idx_to_path[idx] = p
        seen: set[tuple[str, str]] = set()
        for s in node_set:
            for t in edges_out.get(s, ()):
                if t in node_set:
                    key = (min(s, t), max(s, t))
                    if key not in seen:
                        seen.add(key)
                        graph.add_edge(path_to_idx[s], path_to_idx[t], None)

        # Iteratively remove non-seed, non-articulation-point nodes
        current_nodes = set(node_set)
        changed = True
        while changed:
            changed = False
            art_paths = {
                idx_to_path[i]
                for i in rustworkx.articulation_points(graph)
                if i in idx_to_path
            }
            removable = current_nodes - candidate_paths - art_paths
            if removable:
                node = next(iter(removable))
                idx = path_to_idx[node]
                graph.remove_node(idx)
                del path_to_idx[node]
                del idx_to_path[idx]
                current_nodes.discard(node)
                changed = True

        edges_out_frozen = {s: frozenset(ts) for s, ts in edges_out.items()}
        return GroverResult(
            candidates=RustworkxGraph._subgraph_candidates(
                current_nodes, edges_out_frozen, edge_types, "min_meeting_subgraph",
            ),
        )

    # ------------------------------------------------------------------
    # Centrality algorithms
    # ------------------------------------------------------------------

    async def _run_centrality(
        self,
        operation: str,
        rx_fn: Any,
        candidates: GroverResult,
        session: AsyncSession,
        **kwargs: object,
    ) -> GroverResult:
        """Generic centrality: ensure_fresh -> snapshot -> to_thread."""
        try:
            await self.ensure_fresh(session)
            nodes, edges_out = self._snapshot()
            return await asyncio.to_thread(
                self._centrality_impl, nodes, edges_out, candidates, operation, rx_fn, kwargs,
            )

        except Exception as e:
            return GroverResult(success=False, errors=[f"{operation} failed: {e}"])

    @staticmethod
    def _centrality_impl(
        nodes: frozenset[str],
        edges_out: dict[str, frozenset[str]],
        candidates: GroverResult,
        operation: str,
        rx_fn: Any,
        kwargs: dict[str, object],
    ) -> GroverResult:
        graph, _, idx_to_path = RustworkxGraph._build_graph_from(nodes, edges_out)
        if graph.num_nodes() == 0:
            return GroverResult()
        scores = rx_fn(graph, **kwargs)
        raw = {idx_to_path[idx]: score for idx, score in scores.items() if idx in idx_to_path}
        # Filter to candidate paths if any were provided
        candidate_paths = set(RustworkxGraph._extract_paths(candidates))
        if candidate_paths:
            raw = {p: s for p, s in raw.items() if p in candidate_paths}
        return GroverResult(
            candidates=RustworkxGraph._score_candidates(raw, operation),
        )

    async def pagerank(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """PageRank centrality scores."""
        return await self._run_centrality(
            "pagerank", rustworkx.pagerank, candidates, session,
        )

    async def betweenness_centrality(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """Betweenness centrality scores."""
        return await self._run_centrality(
            "betweenness_centrality",
            rustworkx.digraph_betweenness_centrality,
            candidates,
            session,
            normalized=True,
        )

    async def closeness_centrality(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """Closeness centrality scores."""
        return await self._run_centrality(
            "closeness_centrality",
            rustworkx.closeness_centrality,
            candidates,
            session,
        )

    async def degree_centrality(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """Degree centrality (in + out) scores."""
        return await self._run_centrality(
            "degree_centrality",
            rustworkx.digraph_degree_centrality,
            candidates,
            session,
        )

    async def in_degree_centrality(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """In-degree centrality scores."""
        return await self._run_centrality(
            "in_degree_centrality",
            rustworkx.in_degree_centrality,
            candidates,
            session,
        )

    async def out_degree_centrality(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """Out-degree centrality scores."""
        return await self._run_centrality(
            "out_degree_centrality",
            rustworkx.out_degree_centrality,
            candidates,
            session,
        )

    async def hits(
        self,
        candidates: GroverResult,
        *,
        max_iter: int = 1000,
        tol: float = 1e-8,
        session: AsyncSession,
    ) -> GroverResult:
        """HITS hub and authority scores."""
        try:
            await self.ensure_fresh(session)
            nodes, edges_out = self._snapshot()
            return await asyncio.to_thread(
                self._hits_impl, nodes, edges_out, candidates, max_iter, tol,
            )

        except Exception as e:
            return GroverResult(success=False, errors=[f"hits failed: {e}"])

    @staticmethod
    def _hits_impl(
        nodes: frozenset[str],
        edges_out: dict[str, frozenset[str]],
        candidates: GroverResult,
        max_iter: int,
        tol: float,
    ) -> GroverResult:
        graph, _, idx_to_path = RustworkxGraph._build_graph_from(nodes, edges_out)
        candidate_paths = set(RustworkxGraph._extract_paths(candidates))

        if graph.num_nodes() == 0 or graph.num_edges() == 0:
            # Filter to graph-only paths, consistent with the normal path
            # where non-graph candidates are silently dropped via score dicts.
            graph_paths = set(idx_to_path.values())
            all_paths = sorted(candidate_paths & graph_paths) if candidate_paths else sorted(graph_paths)
            return GroverResult(
                candidates=[
                    Candidate(
                        path=p,
                        details=(
                            Detail(
                                operation="hits",
                                score=0.0,
                                metadata={"authority": 0.0, "hub": 0.0},
                            ),
                        ),
                    )
                    for p in all_paths
                ],
            )

        hubs_raw, auths_raw = rustworkx.hits(graph, max_iter=max_iter, tol=tol)
        hubs = {idx_to_path[idx]: score for idx, score in hubs_raw.items() if idx in idx_to_path}
        auths = {idx_to_path[idx]: score for idx, score in auths_raw.items() if idx in idx_to_path}
        if candidate_paths:
            hubs = {p: s for p, s in hubs.items() if p in candidate_paths}
            auths = {p: s for p, s in auths.items() if p in candidate_paths}

        all_paths = sorted(auths, key=auths.__getitem__, reverse=True)
        return GroverResult(
            candidates=[
                Candidate(
                    path=p,
                    details=(
                        Detail(
                            operation="hits",
                            score=auths[p],
                            metadata={
                                "authority": auths[p],
                                "hub": hubs[p],
                            },
                        ),
                    ),
                )
                for p in all_paths
            ],
        )
