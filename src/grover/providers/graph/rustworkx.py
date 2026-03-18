"""RustworkxGraph — rustworkx-backed graph store implementing GraphProvider protocol."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any, NamedTuple

import rustworkx

from grover.models.database.connection import FileConnectionModel, FileConnectionModelBase
from grover.models.internal.evidence import Evidence, GraphCentralityEvidence, GraphRelationshipEvidence
from grover.models.internal.ref import File, FileConnection
from grover.models.internal.results import FileSearchResult, FileSearchSet
from grover.ref import Ref as LegacyRef

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession


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


class RustworkxGraph:
    """Directed knowledge graph over file paths.

    Stores topology as ``_nodes`` (set of paths) and adjacency dicts
    ``_out`` (source → targets) / ``_in`` (target → sources).
    No tuple-per-edge allocation — O(degree) lookups for predecessors
    and successors instead of O(|E|) scans.

    Query/algorithm methods are ``async def``:
    - Light reads run inline (no thread overhead).
    - Heavy algorithms use ``asyncio.to_thread`` with a snapshot for concurrency.

    Mutations stay synchronous (trivial set operations, called from background tasks).

    Implements the ``GraphProvider`` protocol.
    """

    def __init__(self, *, stale_after: float | None = None) -> None:
        self.nodes: set[str] = set()
        self.edges_out: dict[str, set[str]] = {}  # source → targets
        self.edges_in: dict[str, set[str]] = {}  # target → sources
        self.file_connection_model: type[FileConnectionModelBase] = FileConnectionModel
        # Staleness tracking
        self.loaded_at: float | None = None
        self.stale_after: float | None = stale_after

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
        if self.loaded_at is None:
            # Never loaded from SQL — only refresh if the graph is also empty.
            # A non-empty graph was populated via writes (warm from mutations).
            return not self.nodes
        if self.stale_after is None:
            return False  # No TTL — manual only
        return (time.monotonic() - self.loaded_at) > self.stale_after

    def set_connection_model(self, model: type[FileConnectionModelBase]) -> None:
        """Set the connection model used by ``from_sql`` to load edges."""
        self.file_connection_model = model

    async def _ensure_fresh(self, session: AsyncSession) -> None:
        """Load from DB if never loaded or TTL exceeded."""
        if not self.needs_refresh:
            return
        await self.from_sql(session)

    # ------------------------------------------------------------------
    # Graph construction helpers
    # ------------------------------------------------------------------

    def _snapshot(self) -> tuple[frozenset[str], dict[str, frozenset[str]]]:
        """Return immutable copies of nodes and edges_out for thread-safe reads."""
        return (
            frozenset(self.nodes),
            {s: frozenset(ts) for s, ts in self.edges_out.items()},
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

    def _build_graph(self) -> tuple[rustworkx.PyDiGraph, dict[str, int], dict[int, str]]:
        """Build a fresh PyDiGraph from current topology."""
        nodes, edges_out = self._snapshot()
        return self._build_graph_from(nodes, edges_out)

    # ------------------------------------------------------------------
    # Node operations (sync mutations)
    # ------------------------------------------------------------------

    def add_node(self, path: str, **attrs: object) -> None:
        """Add a node. Extra *attrs* are accepted for protocol compat but not stored."""
        if path not in self.nodes:
            self.nodes.add(path)

    def remove_node(self, path: str) -> None:
        """Remove a node and all incident edges. Raises ``KeyError`` if missing."""
        self._require_node(path)
        self.nodes.discard(path)
        # Remove outgoing edges: for each target, remove path from their _in
        for t in self.edges_out.pop(path, set()):
            in_set = self.edges_in.get(t)
            if in_set is not None:
                in_set.discard(path)
                if not in_set:
                    del self.edges_in[t]
        # Remove incoming edges: for each source, remove path from their _out
        for s in self.edges_in.pop(path, set()):
            out_set = self.edges_out.get(s)
            if out_set is not None:
                out_set.discard(path)
                if not out_set:
                    del self.edges_out[s]

    def has_node(self, path: str) -> bool:
        return path in self.nodes

    def get_node(self, path: str) -> dict[str, Any]:
        """Return minimal node data dict. Raises ``KeyError`` if missing."""
        self._require_node(path)
        return {"path": path}

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
        self.nodes.add(source)
        self.nodes.add(target)
        self.edges_out.setdefault(source, set()).add(target)
        self.edges_in.setdefault(target, set()).add(source)

    def remove_edge(self, source: str, target: str) -> None:
        """Remove the edge between *source* and *target*. Raises ``KeyError``."""
        self._require_edge(source, target)
        out_set = self.edges_out.get(source)
        if out_set is not None:
            out_set.discard(target)
            if not out_set:
                del self.edges_out[source]
        in_set = self.edges_in.get(target)
        if in_set is not None:
            in_set.discard(source)
            if not in_set:
                del self.edges_in[target]

    def has_edge(self, source: str, target: str) -> bool:
        return target in self.edges_out.get(source, set())

    def get_edge(self, source: str, target: str) -> dict[str, Any]:
        """Return minimal edge data dict. Raises ``KeyError`` if missing."""
        self._require_edge(source, target)
        return {
            "id": str(uuid.uuid4()),
            "source": source,
            "target": target,
            "type": "",
            "weight": 1.0,
            "metadata": {},
        }

    @property
    def edges(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Return all edges as ``(source, target, data)`` triples."""
        return [
            (s, t, {"id": "", "source": s, "target": t, "type": "", "weight": 1.0, "metadata": {}})
            for s, targets in self.edges_out.items()
            for t in targets
        ]

    # ------------------------------------------------------------------
    # Graph-level properties
    # ------------------------------------------------------------------

    @property
    def graph(self) -> rustworkx.PyDiGraph:
        """Access the underlying rustworkx directed graph."""
        g, _, _ = self._build_graph()
        return g

    def __repr__(self) -> str:
        return f"RustworkxGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_single_path(candidates: FileSearchSet) -> str:
        paths = candidates.paths
        if len(paths) != 1:
            raise ValueError(f"Expected exactly 1 path, got {len(paths)}")
        return paths[0]

    @staticmethod
    def _require_two_paths(candidates: FileSearchSet) -> tuple[str, str]:
        paths = candidates.paths
        if len(paths) != 2:
            raise ValueError(f"Expected exactly 2 paths, got {len(paths)}")
        return paths[0], paths[1]

    # ------------------------------------------------------------------
    # Light reads — async inline (no thread overhead)
    # ------------------------------------------------------------------

    async def predecessors(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        """One-hop backward: nodes with edges pointing to any candidate."""
        await self._ensure_fresh(session)

        query_paths = set(candidates.paths) & self.nodes
        predecessor_targets: dict[str, list[str]] = {}

        for t in query_paths:
            for s in self.edges_in.get(t, set()):
                if s not in query_paths:
                    predecessor_targets.setdefault(s, []).append(t)

        predecessors = sorted(predecessor_targets)
        return FileSearchResult(
            success=True,
            message=f"{len(predecessors)} predecessor(s)",
            files=[
                File(
                    path=p,
                    evidence=[
                        GraphRelationshipEvidence(
                            operation="predecessors",
                            paths=sorted(predecessor_targets[p]),
                        )
                    ],
                )
                for p in predecessors
            ],
        )

    async def successors(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        """One-hop forward: nodes that any candidate points to."""
        await self._ensure_fresh(session)

        query_paths = set(candidates.paths) & self.nodes
        successor_sources: dict[str, list[str]] = {}

        for s in query_paths:
            for t in self.edges_out.get(s, set()):
                if t not in query_paths:
                    successor_sources.setdefault(t, []).append(s)

        successors = sorted(successor_sources)
        return FileSearchResult(
            success=True,
            message=f"{len(successors)} successor(s)",
            files=[
                File(
                    path=p,
                    evidence=[
                        GraphRelationshipEvidence(
                            operation="successors",
                            paths=sorted(successor_sources[p]),
                        )
                    ],
                )
                for p in successors
            ],
        )

    # ------------------------------------------------------------------
    # Heavy algorithms — async + to_thread with snapshot
    # ------------------------------------------------------------------

    # --- Multi-path reachability (ancestors / descendants) ---

    async def ancestors(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        """Transitive backward: union of all ancestor sets, excluding candidates."""
        await self._ensure_fresh(session)
        valid_paths = set(candidates.paths) & self.nodes
        if not valid_paths:
            return FileSearchResult(success=True, message="0 ancestor(s)")
        nodes, edges_out = self._snapshot()
        return await asyncio.to_thread(self._ancestors_impl, nodes, edges_out, valid_paths)

    @staticmethod
    def _ancestors_impl(
        nodes: frozenset[str],
        edges_out: dict[str, frozenset[str]],
        valid_paths: set[str],
    ) -> FileSearchResult:
        graph, path_to_idx, idx_to_path = RustworkxGraph._build_graph_from(nodes, edges_out)
        result_candidates: dict[str, list[str]] = {}
        for candidate in valid_paths:
            for i in rustworkx.ancestors(graph, path_to_idx[candidate]):
                p = idx_to_path.get(i)
                if p is not None and p not in valid_paths:
                    result_candidates.setdefault(p, []).append(candidate)

        result = sorted(result_candidates)
        return FileSearchResult(
            success=True,
            message=f"{len(result)} ancestor(s)",
            files=[
                File(
                    path=p,
                    evidence=[
                        GraphRelationshipEvidence(
                            operation="ancestors",
                            paths=sorted(result_candidates[p]),
                        )
                    ],
                )
                for p in result
            ],
        )

    async def descendants(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        """Transitive forward: union of all descendant sets, excluding candidates."""
        await self._ensure_fresh(session)
        valid_paths = set(candidates.paths) & self.nodes
        if not valid_paths:
            return FileSearchResult(success=True, message="0 descendant(s)")
        nodes, edges_out = self._snapshot()
        return await asyncio.to_thread(self._descendants_impl, nodes, edges_out, valid_paths)

    @staticmethod
    def _descendants_impl(
        nodes: frozenset[str],
        edges_out: dict[str, frozenset[str]],
        valid_paths: set[str],
    ) -> FileSearchResult:
        graph, path_to_idx, idx_to_path = RustworkxGraph._build_graph_from(nodes, edges_out)
        result_candidates: dict[str, list[str]] = {}
        for candidate in valid_paths:
            for i in rustworkx.descendants(graph, path_to_idx[candidate]):
                p = idx_to_path.get(i)
                if p is not None and p not in valid_paths:
                    result_candidates.setdefault(p, []).append(candidate)

        result = sorted(result_candidates)
        return FileSearchResult(
            success=True,
            message=f"{len(result)} descendant(s)",
            files=[
                File(
                    path=p,
                    evidence=[
                        GraphRelationshipEvidence(
                            operation="descendants",
                            paths=sorted(result_candidates[p]),
                        )
                    ],
                )
                for p in result
            ],
        )

    # --- Neighborhood ---

    async def neighborhood(
        self,
        candidates: FileSearchSet,
        *,
        max_depth: int = 2,
        session: AsyncSession,
    ) -> FileSearchResult:
        """Bounded undirected BFS around a single node."""
        path = self._require_single_path(candidates)
        await self._ensure_fresh(session)
        if path not in self.nodes:
            return FileSearchResult(success=True, message="0 node(s), 0 edge(s)")

        # BFS
        visited: set[str] = {path}
        frontier: set[str] = {path}
        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for node in frontier:
                for n in self.edges_out.get(node, ()):
                    if n not in visited:
                        visited.add(n)
                        next_frontier.add(n)
                for n in self.edges_in.get(node, ()):
                    if n not in visited:
                        visited.add(n)
                        next_frontier.add(n)
            frontier = next_frontier
            if not frontier:
                break

        # Build induced subgraph from visited nodes
        ev: list[Evidence] = [GraphRelationshipEvidence(operation="neighborhood", paths=[path])]
        connections: list[FileConnection] = []
        for s in visited:
            connections.extend(
                FileConnection(path=f"{s}[]{t}", source_path=s, target_path=t, type="", weight=1.0, evidence=ev)
                for t in self.edges_out.get(s, ())
                if t in visited
            )
        files = [File(path=p, evidence=ev) for p in sorted(visited)]
        return FileSearchResult(
            success=True,
            message=f"{len(files)} node(s), {len(connections)} edge(s)",
            files=files,
            connections=connections,
        )

    # --- Subgraph building helper ---

    @staticmethod
    def _build_subgraph_result(
        node_set: set[str],
        edges_out: dict[str, frozenset[str]],
        operation: str,
    ) -> tuple[list[File], list[FileConnection], str]:
        """Build File list, FileConnection list, and message from node/edge sets."""
        ev: list[Evidence] = [GraphRelationshipEvidence(operation=operation)]
        connections: list[FileConnection] = []
        for s in node_set:
            connections.extend(
                FileConnection(path=f"{s}[]{t}", source_path=s, target_path=t, type="", weight=1.0, evidence=ev)
                for t in edges_out.get(s, ())
                if t in node_set
            )
        files = [File(path=n, evidence=ev) for n in sorted(node_set)]
        message = f"{len(files)} node(s), {len(connections)} edge(s)"
        return files, connections, message

    # --- Candidate augmentation: inject unknown paths with inferred edges ---

    @staticmethod
    def _augment_with_candidates(
        nodes: frozenset[str],
        edges_out: dict[str, frozenset[str]],
        candidate_paths: list[str],
    ) -> tuple[frozenset[str], dict[str, frozenset[str]]]:
        """Add candidate paths to graph, inferring edges for chunks/versions.

        - **Chunk** (``/a.py#login``): infer ``(/a.py, /a.py#login)`` edge
        - **Version** (``/a.py@3``): infer ``(/a.py, /a.py@3)`` edge
        - **Plain file**: add as isolated node (no inferred edge)
        """
        extra_nodes: set[str] = set()
        extra_edges: dict[str, set[str]] = {}
        for p in candidate_paths:
            if p not in nodes:
                extra_nodes.add(p)
                ref = LegacyRef(path=p)
                if ref.is_chunk or ref.is_version:
                    base = ref.base_path
                    extra_nodes.add(base)
                    extra_edges.setdefault(base, set()).add(p)
        if not extra_nodes:
            return nodes, edges_out
        merged = dict(edges_out)
        for s, ts in extra_edges.items():
            existing = merged.get(s, frozenset())
            merged[s] = existing | frozenset(ts)
        return nodes | frozenset(extra_nodes), merged

    # --- Meeting subgraph ---

    async def meeting_subgraph(
        self,
        candidates: FileSearchSet,
        *,
        session: AsyncSession,
    ) -> FileSearchResult:
        """Find minimal subgraph connecting all candidate nodes.

        Uses multi-source BFS with union-find to detect when seed
        wavefronts meet, then leaf-strips non-seed nodes. O(V+E).
        """
        await self._ensure_fresh(session)
        valid_seeds = [p for p in candidates.paths if p in self.nodes]
        if len(valid_seeds) <= 1:
            ev: list[Evidence] = [GraphRelationshipEvidence(operation="meeting_subgraph", paths=valid_seeds)]
            files = [File(path=p, evidence=ev) for p in valid_seeds]
            return FileSearchResult(
                success=True,
                message=f"{len(files)} node(s), 0 edge(s)",
                files=files,
            )
        _, edges_out = self._snapshot()
        edges_in = {t: frozenset(ss) for t, ss in self.edges_in.items()}
        return await asyncio.to_thread(self._meeting_subgraph_impl, edges_out, edges_in, valid_seeds)

    @staticmethod
    def _meeting_subgraph_impl(
        edges_out: dict[str, frozenset[str]],
        edges_in: dict[str, frozenset[str]],
        seeds: list[str],
    ) -> FileSearchResult:
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

        # Leaf stripping — remove non-seed leaves iteratively O(n+e)
        kept = RustworkxGraph._strip_leaves(kept, edges_out, edges_in, seed_set)

        # Build result
        ev: list[Evidence] = [GraphRelationshipEvidence(operation="meeting_subgraph")]
        connections: list[FileConnection] = []
        for s in kept:
            connections.extend(
                FileConnection(path=f"{s}[]{t}", source_path=s, target_path=t, type="", weight=1.0, evidence=ev)
                for t in edges_out.get(s, ())
                if t in kept
            )
        files = [File(path=p, evidence=ev) for p in sorted(kept)]
        return FileSearchResult(
            success=True,
            message=f"{len(files)} node(s), {len(connections)} edge(s)",
            files=files,
            connections=connections,
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
            s_targets = set()
            for t in edges_out.get(s, ()):
                if t in kept:
                    s_targets.add(t)
            succs[s] = s_targets
        for t in kept:
            t_sources = set()
            for s in edges_in.get(t, ()):
                if s in kept:
                    t_sources.add(s)
            preds[t] = t_sources

        queue = [n for n in kept if n not in protected and (not succs.get(n) or not preds.get(n))]
        removed: set[str] = set()
        while queue:
            node = queue.pop()
            if node in removed or node in protected:
                continue
            removed.add(node)
            for succ in succs.get(node, ()):
                preds[succ].discard(node)
                if succ not in protected and succ not in removed and (not preds[succ] or not succs[succ]):
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

    # --- Min meeting subgraph ---

    async def min_meeting_subgraph(
        self,
        candidates: FileSearchSet,
        *,
        session: AsyncSession,
    ) -> FileSearchResult:
        """Pruned meeting subgraph — drop non-candidate nodes while staying connected.

        Uses ``rustworkx.articulation_points`` to identify nodes whose removal
        would disconnect the graph.  Non-seed, non-articulation-point nodes are
        removed one at a time (recomputing articulation points after each
        removal) until only seeds and structurally critical intermediaries
        remain.
        """
        meeting = await self.meeting_subgraph(candidates, session=session)
        if len(meeting.files) <= len(candidates.files):
            return meeting

        candidate_paths = set(candidates.paths)
        node_set = {f.path for f in meeting.files}
        edges_out: dict[str, set[str]] = {}
        for c in meeting.connections:
            edges_out.setdefault(c.source_path, set()).add(c.target_path)

        return await asyncio.to_thread(self._min_meeting_impl, node_set, edges_out, candidate_paths)

    @staticmethod
    def _min_meeting_impl(
        node_set: set[str],
        edges_out: dict[str, set[str]],
        candidate_paths: set[str],
    ) -> FileSearchResult:
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
            art_paths = {idx_to_path[i] for i in rustworkx.articulation_points(graph) if i in idx_to_path}
            protected = candidate_paths | art_paths
            removable = current_nodes - protected
            if removable:
                node = next(iter(removable))
                idx = path_to_idx[node]
                graph.remove_node(idx)
                del path_to_idx[node]
                del idx_to_path[idx]
                current_nodes.discard(node)
                changed = True

        edges_out_frozen = {s: frozenset(ts) for s, ts in edges_out.items()}
        files, connections, message = RustworkxGraph._build_subgraph_result(
            current_nodes, edges_out_frozen, "min_meeting_subgraph"
        )
        return FileSearchResult(success=True, message=message, files=files, connections=connections)

    # ------------------------------------------------------------------
    # Centrality algorithms
    # ------------------------------------------------------------------

    @staticmethod
    def _scores_to_candidates(scores: dict[str, float], operation: str, algorithm: str) -> list[File]:
        """Convert a {path: score} dict to sorted File list."""
        sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            File(
                path=path,
                evidence=[GraphCentralityEvidence(operation=algorithm, score=score)],
            )
            for path, score in sorted_items
        ]

    # --- Resolve topology for centrality from FileSearchSet ---

    @staticmethod
    def _resolve_graph_from(
        nodes: frozenset[str],
        edges_out: dict[str, frozenset[str]],
        candidates: FileSearchSet,
    ) -> tuple[rustworkx.PyDiGraph, dict[str, int], dict[int, str]]:
        """Thread-safe resolve: build the appropriate graph from candidates.

        - If candidates has connections, use that topology directly.
        - If candidates has only files (no connections), use full graph.
        - If candidates is empty, use the full graph.
        """
        if not candidates.files:
            return RustworkxGraph._build_graph_from(nodes, edges_out)
        if candidates.connections:
            sub_nodes = frozenset(f.path for f in candidates.files)
            sub_edges_out: dict[str, frozenset[str]] = {}
            for cc in candidates.connections:
                s, t = cc.source_path, cc.target_path
                existing = sub_edges_out.get(s, frozenset())
                sub_edges_out[s] = existing | frozenset([t])
            return RustworkxGraph._build_graph_from(sub_nodes, sub_edges_out)
        # Files only — use full graph topology
        return RustworkxGraph._build_graph_from(nodes, edges_out)

    # --- Generic centrality dispatcher ---

    class _CentralitySpec(NamedTuple):
        operation: str
        algorithm: str

    async def _run_centrality(
        self,
        spec: _CentralitySpec,
        rx_fn: Callable[..., Any],
        candidates: FileSearchSet,
        session: AsyncSession,
        **kwargs: object,
    ) -> FileSearchResult:
        """Generic centrality: ensure_fresh → snapshot → to_thread."""
        await self._ensure_fresh(session)
        nodes, edges_out = self._snapshot()
        return await asyncio.to_thread(self._centrality_impl, nodes, edges_out, candidates, spec, rx_fn, kwargs)

    @staticmethod
    def _centrality_impl(
        nodes: frozenset[str],
        edges_out: dict[str, frozenset[str]],
        candidates: FileSearchSet,
        spec: RustworkxGraph._CentralitySpec,
        rx_fn: Callable[..., Any],
        kwargs: dict[str, object],
    ) -> FileSearchResult:
        graph, _, idx_to_path = RustworkxGraph._resolve_graph_from(nodes, edges_out, candidates)
        if graph.num_nodes() == 0:
            return FileSearchResult(success=True, message="0 node(s)")
        scores = rx_fn(graph, **kwargs)
        raw = {idx_to_path[idx]: score for idx, score in scores.items() if idx in idx_to_path}
        # Filter to only candidate paths if candidates were provided
        candidate_paths = set(candidates.paths)
        if candidate_paths:
            raw = {p: s for p, s in raw.items() if p in candidate_paths}
        fcs = RustworkxGraph._scores_to_candidates(raw, spec.operation, spec.algorithm)
        return FileSearchResult(success=True, message=f"{len(fcs)} node(s)", files=fcs)

    # --- PageRank ---

    async def pagerank(
        self,
        candidates: FileSearchSet,
        *,
        alpha: float = 0.85,
        personalization: dict[str, float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        session: AsyncSession,
    ) -> FileSearchResult:
        """PageRank centrality scores."""
        await self._ensure_fresh(session)
        nodes, edges_out = self._snapshot()
        return await asyncio.to_thread(
            self._pagerank_impl, nodes, edges_out, candidates, alpha, personalization, max_iter, tol
        )

    @staticmethod
    def _pagerank_impl(
        nodes: frozenset[str],
        edges_out: dict[str, frozenset[str]],
        candidates: FileSearchSet,
        alpha: float,
        personalization: dict[str, float] | None,
        max_iter: int,
        tol: float,
    ) -> FileSearchResult:
        graph, path_to_idx, idx_to_path = RustworkxGraph._resolve_graph_from(nodes, edges_out, candidates)
        if graph.num_nodes() == 0:
            return FileSearchResult(success=True, message="0 node(s)")
        pers = None
        if personalization:
            pers = {path_to_idx[p]: w for p, w in personalization.items() if p in path_to_idx}
            if not pers:
                pers = None
        scores = rustworkx.pagerank(graph, alpha=alpha, personalization=pers, max_iter=max_iter, tol=tol)
        raw = {idx_to_path[idx]: score for idx, score in scores.items() if idx in idx_to_path}
        # Filter to only candidate paths if candidates were provided
        candidate_paths = set(candidates.paths)
        if candidate_paths:
            raw = {p: s for p, s in raw.items() if p in candidate_paths}
        fcs = RustworkxGraph._scores_to_candidates(raw, "pagerank", "pagerank")
        return FileSearchResult(
            success=True,
            message=f"{len(fcs)} node(s)",
            files=fcs,
        )

    # --- Betweenness ---

    async def betweenness_centrality(
        self,
        candidates: FileSearchSet,
        *,
        normalized: bool = True,
        session: AsyncSession,
    ) -> FileSearchResult:
        """Betweenness centrality scores."""
        return await self._run_centrality(
            self._CentralitySpec("betweenness_centrality", "betweenness_centrality"),
            rustworkx.digraph_betweenness_centrality,
            candidates,
            session,
            normalized=normalized,
        )

    # --- Closeness ---

    async def closeness_centrality(
        self,
        candidates: FileSearchSet,
        *,
        session: AsyncSession,
    ) -> FileSearchResult:
        """Closeness centrality scores."""
        return await self._run_centrality(
            self._CentralitySpec("closeness_centrality", "closeness_centrality"),
            rustworkx.closeness_centrality,
            candidates,
            session,
        )

    # --- HITS ---

    async def hits(
        self,
        candidates: FileSearchSet,
        *,
        max_iter: int = 100,
        tol: float = 1e-8,
        session: AsyncSession,
    ) -> FileSearchResult:
        """HITS hub and authority scores."""
        await self._ensure_fresh(session)
        nodes, edges_out = self._snapshot()
        return await asyncio.to_thread(self._hits_impl, nodes, edges_out, candidates, max_iter, tol)

    @staticmethod
    def _hits_impl(
        nodes: frozenset[str],
        edges_out: dict[str, frozenset[str]],
        candidates: FileSearchSet,
        max_iter: int,
        tol: float,
    ) -> FileSearchResult:
        graph, path_to_idx, idx_to_path = RustworkxGraph._resolve_graph_from(nodes, edges_out, candidates)
        candidate_paths = set(candidates.paths)
        if graph.num_nodes() == 0 or graph.num_edges() == 0:
            all_paths = sorted(candidate_paths if candidate_paths else path_to_idx)
            return FileSearchResult(
                success=True,
                message=f"HITS computed for {len(all_paths)} node(s)",
                files=[
                    File(
                        path=p,
                        evidence=[
                            GraphCentralityEvidence(operation="hits", score=0.0, scores={"authority": 0.0, "hub": 0.0}),
                        ],
                    )
                    for p in all_paths
                ],
            )
        hubs_raw, auths_raw = rustworkx.hits(graph, max_iter=max_iter, tol=tol)
        hubs = {idx_to_path[idx]: score for idx, score in hubs_raw.items() if idx in idx_to_path}
        auths = {idx_to_path[idx]: score for idx, score in auths_raw.items() if idx in idx_to_path}
        # Filter to only candidate paths if candidates were provided
        if candidate_paths:
            hubs = {p: s for p, s in hubs.items() if p in candidate_paths}
            auths = {p: s for p, s in auths.items() if p in candidate_paths}
        all_paths = sorted(set(hubs) | set(auths), key=lambda p: auths.get(p, 0.0), reverse=True)
        return FileSearchResult(
            success=True,
            message=f"HITS computed for {len(all_paths)} node(s)",
            files=[
                File(
                    path=p,
                    evidence=[
                        GraphCentralityEvidence(
                            operation="hits",
                            score=auths.get(p, 0.0),
                            scores={
                                "authority": auths.get(p, 0.0),
                                "hub": hubs.get(p, 0.0),
                            },
                        ),
                    ],
                )
                for p in all_paths
            ],
        )

    # --- Katz ---

    async def katz_centrality(
        self,
        candidates: FileSearchSet,
        *,
        alpha: float = 0.1,
        beta: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
        session: AsyncSession,
    ) -> FileSearchResult:
        """Katz centrality scores."""
        return await self._run_centrality(
            self._CentralitySpec("katz_centrality", "katz_centrality"),
            rustworkx.katz_centrality,
            candidates,
            session,
            alpha=alpha,
            beta=beta,
            max_iter=max_iter,
            tol=tol,
        )

    # --- Degree centrality ---

    async def degree_centrality(
        self,
        candidates: FileSearchSet,
        *,
        session: AsyncSession,
    ) -> FileSearchResult:
        """Degree centrality (in + out) scores."""
        return await self._run_centrality(
            self._CentralitySpec("degree_centrality", "degree_centrality"),
            rustworkx.digraph_degree_centrality,
            candidates,
            session,
        )

    async def in_degree_centrality(
        self,
        candidates: FileSearchSet,
        *,
        session: AsyncSession,
    ) -> FileSearchResult:
        """In-degree centrality scores."""
        return await self._run_centrality(
            self._CentralitySpec("in_degree_centrality", "in_degree_centrality"),
            rustworkx.in_degree_centrality,
            candidates,
            session,
        )

    async def out_degree_centrality(
        self,
        candidates: FileSearchSet,
        *,
        session: AsyncSession,
    ) -> FileSearchResult:
        """Out-degree centrality scores."""
        return await self._run_centrality(
            self._CentralitySpec("out_degree_centrality", "out_degree_centrality"),
            rustworkx.out_degree_centrality,
            candidates,
            session,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def from_sql(self, session: AsyncSession) -> None:
        """Load graph state from the database, replacing in-memory state.

        Only nodes that participate in connections are loaded — files with no
        connections are not added to the graph.

        Build-then-swap: new state is assembled in local variables and assigned
        atomically at the end, so concurrent readers never see an empty graph.
        """
        from sqlalchemy import select

        new_nodes: set[str] = set()
        new_out: dict[str, set[str]] = {}
        new_in: dict[str, set[str]] = {}

        result = await session.execute(select(self.file_connection_model))
        for edge_row in result.scalars().all():
            new_nodes.add(edge_row.source_path)
            new_nodes.add(edge_row.target_path)
            new_out.setdefault(edge_row.source_path, set()).add(edge_row.target_path)
            new_in.setdefault(edge_row.target_path, set()).add(edge_row.source_path)

        # Atomic swap
        self.nodes = new_nodes
        self.edges_out = new_out
        self.edges_in = new_in
        self.loaded_at = time.monotonic()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_node(self, path: str) -> None:
        """Raise ``KeyError`` if *path* is not in the graph."""
        if path not in self.nodes:
            msg = f"Node not found: {path!r}"
            raise KeyError(msg)

    def _require_edge(self, source: str, target: str) -> None:
        """Raise ``KeyError`` if the edge *source* → *target* doesn't exist."""
        self._require_node(source)
        self._require_node(target)
        if target not in self.edges_out.get(source, set()):
            msg = f"No edge from {source!r} to {target!r}"
            raise KeyError(msg)
