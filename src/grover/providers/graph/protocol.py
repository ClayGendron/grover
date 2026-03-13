"""Graph protocol — unified runtime-checkable interface for graph backends.

All graph capabilities (centrality, traversal, subgraph extraction, filtering,
persistence) are part of a single ``GraphProvider`` protocol.
``RustworkxGraph`` is the sole implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.internal.ref import Ref
    from grover.models.internal.results import FileSearchResult


@runtime_checkable
class GraphProvider(Protocol):
    """Graph interface — nodes are file paths, edges are dependencies.

    Mutations are synchronous. Query/algorithm methods are async.
    All async query methods require a ``session`` parameter so the
    provider can self-refresh from the database when stale.
    """

    # ------------------------------------------------------------------
    # Graph Internal Operations (sync mutations + sync utilities)
    # ------------------------------------------------------------------

    def add_node(self, path: str, **attrs: object) -> None: ...

    def remove_node(self, path: str) -> None: ...

    def has_node(self, path: str) -> bool: ...

    def get_node(self, path: str) -> dict: ...

    def nodes(self) -> list[str]: ...

    def add_edge(self, source: str, target: str, edge_type: str, **attrs: object) -> None: ...

    def remove_edge(self, source: str, target: str) -> None: ...

    def has_edge(self, source: str, target: str) -> bool: ...

    def get_edge(self, source: str, target: str) -> dict: ...

    def edges(self) -> list[tuple[str, str, dict]]: ...

    @property
    def node_count(self) -> int: ...

    @property
    def edge_count(self) -> int: ...

    @property
    def graph(self) -> Any: ...

    def remove_file_subgraph(self, path: str) -> list[str]: ...

    async def from_sql(self, session: AsyncSession) -> None: ...

    # ------------------------------------------------------------------
    # Graph APIs — async typed result returns
    # ------------------------------------------------------------------

    # Light reads (async inline)

    async def predecessors(self, path: str, *, session: AsyncSession) -> FileSearchResult: ...

    async def successors(self, path: str, *, session: AsyncSession) -> FileSearchResult: ...

    async def contains(self, path: str, *, session: AsyncSession) -> list[Ref]: ...

    async def by_parent(self, parent_path: str, *, session: AsyncSession) -> list[Ref]: ...

    async def subgraph(self, paths: list[str], *, session: AsyncSession) -> FileSearchResult: ...

    async def neighborhood(
        self,
        path: str,
        *,
        max_depth: int = 2,
        direction: str = "both",
        edge_types: list[str] | None = None,
        session: AsyncSession,
    ) -> FileSearchResult: ...

    async def connecting_subgraph(
        self, paths: list[str], *, session: AsyncSession
    ) -> GraphProvider: ...

    async def common_neighbors(
        self, path1: str, path2: str, *, session: AsyncSession
    ) -> FileSearchResult: ...

    async def node_similarity(
        self,
        path1: str,
        path2: str,
        *,
        method: str = "jaccard",
        session: AsyncSession,
    ) -> float: ...

    async def similar_nodes(
        self,
        path: str,
        *,
        method: str = "jaccard",
        k: int = 10,
        session: AsyncSession,
    ) -> list[tuple[str, float]]: ...

    # Heavy algorithms (async + to_thread)

    async def path_between(
        self, source: str, target: str, *, session: AsyncSession
    ) -> FileSearchResult: ...

    async def ancestors(self, path: str, *, session: AsyncSession) -> FileSearchResult: ...

    async def descendants(self, path: str, *, session: AsyncSession) -> FileSearchResult: ...

    async def has_path(
        self, source: str, target: str, *, session: AsyncSession
    ) -> FileSearchResult: ...

    async def all_simple_paths(
        self,
        source: str,
        target: str,
        *,
        cutoff: int | None = None,
        session: AsyncSession,
    ) -> list[list[str]]: ...

    async def topological_sort(self, *, session: AsyncSession) -> list[str]: ...

    async def shortest_path_length(
        self, source: str, target: str, *, session: AsyncSession
    ) -> float | None: ...

    async def meeting_subgraph(
        self, start_paths: list[str], *, max_size: int = 50, session: AsyncSession
    ) -> FileSearchResult: ...

    async def common_reachable(
        self, paths: list[str], *, direction: str = "forward", session: AsyncSession
    ) -> set[str]: ...

    # Centrality algorithms

    async def pagerank(
        self,
        candidates: FileSearchResult | None = None,
        *,
        alpha: float = 0.85,
        personalization: dict[str, float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        session: AsyncSession,
    ) -> FileSearchResult: ...

    async def betweenness_centrality(
        self,
        candidates: FileSearchResult | None = None,
        *,
        normalized: bool = True,
        session: AsyncSession,
    ) -> FileSearchResult: ...

    async def closeness_centrality(
        self,
        candidates: FileSearchResult | None = None,
        *,
        session: AsyncSession,
    ) -> FileSearchResult: ...

    async def harmonic_centrality(
        self,
        candidates: FileSearchResult | None = None,
        *,
        session: AsyncSession,
    ) -> FileSearchResult: ...

    async def hits(
        self,
        candidates: FileSearchResult | None = None,
        *,
        max_iter: int = 100,
        tol: float = 1e-8,
        session: AsyncSession,
    ) -> FileSearchResult: ...

    async def katz_centrality(
        self,
        candidates: FileSearchResult | None = None,
        *,
        alpha: float = 0.1,
        beta: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
        session: AsyncSession,
    ) -> FileSearchResult: ...

    async def degree_centrality(
        self,
        candidates: FileSearchResult | None = None,
        *,
        session: AsyncSession,
    ) -> FileSearchResult: ...

    async def in_degree_centrality(
        self,
        candidates: FileSearchResult | None = None,
        *,
        session: AsyncSession,
    ) -> FileSearchResult: ...

    async def out_degree_centrality(
        self,
        candidates: FileSearchResult | None = None,
        *,
        session: AsyncSession,
    ) -> FileSearchResult: ...

    # Connectivity

    async def weakly_connected_components(self, *, session: AsyncSession) -> list[set[str]]: ...

    async def strongly_connected_components(self, *, session: AsyncSession) -> list[set[str]]: ...

    async def is_weakly_connected(self, *, session: AsyncSession) -> bool: ...
