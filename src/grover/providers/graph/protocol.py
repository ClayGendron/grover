"""Graph protocol — unified runtime-checkable interface for graph backends.

All graph capabilities (centrality, traversal, subgraph extraction, filtering,
persistence) are part of a single ``GraphProvider`` protocol.
``RustworkxGraph`` is the sole implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.ref import Ref
    from grover.results.search import (
        AncestorsResult,
        BetweennessResult,
        ClosenessResult,
        CommonNeighborsResult,
        DegreeResult,
        DescendantsResult,
        EgoGraphResult,
        FileSearchResult,
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


@runtime_checkable
class GraphProvider(Protocol):
    """Graph interface — nodes are file paths, edges are dependencies.

    Includes all graph capabilities: CRUD, queries, centrality, traversal,
    subgraph extraction, filtering, node similarity, and SQL persistence.
    """

    # ------------------------------------------------------------------
    # Graph Internal Operations
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

    async def from_sql(
        self, session: AsyncSession, file_model: type | None = None, *, path_prefix: str = ""
    ) -> None: ...

    # ------------------------------------------------------------------
    # Graph APIs — typed result returns
    # ------------------------------------------------------------------

    def predecessors(self, path: str) -> PredecessorsResult: ...

    def successors(self, path: str) -> SuccessorsResult: ...

    def path_between(self, source: str, target: str) -> ShortestPathResult: ...

    def contains(self, path: str) -> list[Ref]: ...

    def by_parent(self, parent_path: str) -> list[Ref]: ...

    def remove_file_subgraph(self, path: str) -> list[str]: ...

    # Traversal

    def ancestors(self, path: str) -> AncestorsResult: ...

    def descendants(self, path: str) -> DescendantsResult: ...

    def has_path(self, source: str, target: str) -> HasPathResult: ...

    def all_simple_paths(
        self, source: str, target: str, *, cutoff: int | None = None
    ) -> list[list[str]]: ...

    def topological_sort(self) -> list[str]: ...

    def shortest_path_length(self, source: str, target: str) -> float | None: ...

    # Subgraph extraction

    def subgraph(self, paths: list[str]) -> SubgraphSearchResult: ...

    def neighborhood(
        self,
        path: str,
        *,
        max_depth: int = 2,
        direction: str = "both",
        edge_types: list[str] | None = None,
    ) -> EgoGraphResult: ...

    def meeting_subgraph(
        self, start_paths: list[str], *, max_size: int = 50
    ) -> MeetingSubgraphResult: ...

    def connecting_subgraph(self, paths: list[str]) -> GraphProvider: ...

    def common_reachable(self, paths: list[str], *, direction: str = "forward") -> set[str]: ...

    def common_neighbors(self, path1: str, path2: str) -> CommonNeighborsResult: ...

    # Centrality algorithms — return typed results

    def pagerank(
        self,
        candidates: FileSearchResult | None = None,
        *,
        alpha: float = 0.85,
        personalization: dict[str, float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> PageRankResult: ...

    def betweenness_centrality(
        self,
        candidates: FileSearchResult | None = None,
        *,
        normalized: bool = True,
    ) -> BetweennessResult: ...

    def closeness_centrality(
        self,
        candidates: FileSearchResult | None = None,
    ) -> ClosenessResult: ...

    def harmonic_centrality(
        self,
        candidates: FileSearchResult | None = None,
    ) -> HarmonicResult: ...

    def hits(
        self,
        candidates: FileSearchResult | None = None,
        *,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> HitsResult: ...

    def katz_centrality(
        self,
        candidates: FileSearchResult | None = None,
        *,
        alpha: float = 0.1,
        beta: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> KatzResult: ...

    def degree_centrality(
        self,
        candidates: FileSearchResult | None = None,
    ) -> DegreeResult: ...

    def in_degree_centrality(
        self,
        candidates: FileSearchResult | None = None,
    ) -> DegreeResult: ...

    def out_degree_centrality(
        self,
        candidates: FileSearchResult | None = None,
    ) -> DegreeResult: ...

    # Connectivity

    def weakly_connected_components(self) -> list[set[str]]: ...

    def strongly_connected_components(self) -> list[set[str]]: ...

    def is_weakly_connected(self) -> bool: ...

    # Node similarity

    def node_similarity(self, path1: str, path2: str, *, method: str = "jaccard") -> float: ...

    def similar_nodes(
        self, path: str, *, method: str = "jaccard", k: int = 10
    ) -> list[tuple[str, float]]: ...
