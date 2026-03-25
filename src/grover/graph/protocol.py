"""GraphProvider — runtime-checkable protocol for graph backends.

Defines the interface that all graph implementations must satisfy.
Mutations are synchronous (called from background indexing tasks).
Query/algorithm methods are async (called from the routing layer).

``RustworkxGraph`` is the sole implementation today.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.results import GroverResult


@runtime_checkable
class GraphProvider(Protocol):
    """Graph interface — nodes are file paths, edges are dependencies.

    Mutations are synchronous. Query/algorithm methods are async.
    All async query methods require a ``session`` parameter so the
    provider can self-refresh from the database when stale.

    Query methods accept ``GroverResult`` as their candidates argument
    for composability — the output of one graph operation can feed
    directly into another.
    """

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    async def add_node(self, path: str, *, session: AsyncSession) -> None: ...

    async def remove_node(self, path: str, *, session: AsyncSession) -> None: ...

    async def has_node(self, path: str, *, session: AsyncSession) -> bool: ...

    async def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        *,
        weight: float = 1.0,
        session: AsyncSession,
    ) -> None: ...

    async def remove_edge(self, source: str, target: str, *, session: AsyncSession) -> None: ...

    async def has_edge(self, source: str, target: str, *, session: AsyncSession) -> bool: ...

    @property
    def nodes(self) -> set[str]: ...

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def ensure_fresh(self, session: AsyncSession) -> None: ...

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    async def predecessors(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult: ...

    async def successors(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult: ...

    async def ancestors(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult: ...

    async def descendants(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult: ...

    async def neighborhood(
        self,
        candidates: GroverResult,
        *,
        depth: int = 2,
        session: AsyncSession,
    ) -> GroverResult: ...

    # ------------------------------------------------------------------
    # Subgraph
    # ------------------------------------------------------------------

    async def meeting_subgraph(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult: ...

    async def min_meeting_subgraph(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult: ...

    # ------------------------------------------------------------------
    # Centrality algorithms
    # ------------------------------------------------------------------

    async def pagerank(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult: ...

    async def betweenness_centrality(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult: ...

    async def closeness_centrality(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult: ...

    async def degree_centrality(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult: ...

    async def in_degree_centrality(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult: ...

    async def out_degree_centrality(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession,
    ) -> GroverResult: ...

    async def hits(
        self,
        candidates: GroverResult,
        *,
        max_iter: int = 1000,
        tol: float = 1e-8,
        session: AsyncSession,
    ) -> GroverResult: ...
