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
from collections import deque
from typing import TYPE_CHECKING, Any

import rustworkx

from sqlalchemy import select

from grover.paths import connection_path, parse_kind
from grover.results import Candidate, Detail, GroverResult

if TYPE_CHECKING:
    from collections.abc import Callable

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

    def __init__(self, model: type[GroverObjectBase]) -> None:
        self._model = model
        self._nodes: set[str] = set()
        self._out: dict[str, set[str]] = {}  # source → targets
        self._in: dict[str, set[str]] = {}  # target → sources
        self._edge_types: dict[tuple[str, str], str] = {}  # (source, target) → type
        self._loaded: bool = False

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

    def __repr__(self) -> str:
        edge_count = sum(len(ts) for ts in self._out.values())
        return f"RustworkxGraph(nodes={len(self._nodes)}, edges={edge_count})"

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
        """Load from DB if not yet loaded."""
        if self._loaded:
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

        stmt = select(self._model).where(self._model.kind == "connection")  # type: ignore[arg-type]
        result = await session.execute(stmt)
        for obj in result.scalars().all():
            src = obj.source_path
            tgt = obj.target_path
            if src and tgt:
                new_nodes.add(src)
                new_nodes.add(tgt)
                new_out.setdefault(src, set()).add(tgt)
                new_in.setdefault(tgt, set()).add(src)
                new_edge_types[(src, tgt)] = obj.connection_type or ""

        # Atomic swap
        self._nodes = new_nodes
        self._out = new_out
        self._in = new_in
        self._edge_types = new_edge_types
        self._loaded = True

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
                        metadata={"paths": sorted(paths_dict[p])},  # type: ignore[arg-type]
                    ),
                ),
            )
            for p in sorted(paths_dict)
        ]

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
    def _subgraph_candidates(
        node_set: set[str],
        edges_out: dict[str, frozenset[str]],
        edge_types: dict[tuple[str, str], str],
        operation: str,
    ) -> list[Candidate]:
        """Build node + connection candidates from a subgraph."""
        detail = Detail(operation=operation)
        candidates: list[Candidate] = []

        # Nodes
        for p in sorted(node_set):
            candidates.append(Candidate(path=p, details=(detail,)))

        # Edges as connection candidates
        for s in node_set:
            for t in edges_out.get(s, frozenset()):
                if t in node_set:
                    candidates.append(
                        Candidate(
                            path=connection_path(s, t, edge_types[(s, t)]),
                            weight=1.0,
                            details=(detail,),
                        ),
                    )

        return candidates

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

    async def successors(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult:
        """One-hop forward: nodes that any candidate points to."""
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
