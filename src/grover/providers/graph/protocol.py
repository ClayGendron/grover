"""Graph protocol — unified runtime-checkable interface for graph backends.

All graph capabilities (centrality, traversal, subgraph extraction, filtering,
persistence) are part of a single ``GraphProvider`` protocol.
``RustworkxGraph`` is the sole implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.providers.graph.types import SubgraphResult
    from grover.ref import Ref


@runtime_checkable
class GraphProvider(Protocol):
    """Graph interface — nodes are file paths, edges are dependencies.

    Includes all graph capabilities: CRUD, queries, centrality, traversal,
    subgraph extraction, filtering, node similarity, and SQL persistence.
    """

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, path: str, **attrs: object) -> None: ...

    def remove_node(self, path: str) -> None: ...

    def has_node(self, path: str) -> bool: ...

    def get_node(self, path: str) -> dict: ...

    def nodes(self) -> list[str]: ...

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, source: str, target: str, edge_type: str, **attrs: object) -> None: ...

    def remove_edge(self, source: str, target: str) -> None: ...

    def has_edge(self, source: str, target: str) -> bool: ...

    def get_edge(self, source: str, target: str) -> dict: ...

    def edges(self) -> list[tuple[str, str, dict]]: ...

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def dependents(self, path: str) -> list[Ref]: ...

    def dependencies(self, path: str) -> list[Ref]: ...

    def impacts(self, path: str, max_depth: int = 3) -> list[Ref]: ...

    def path_between(self, source: str, target: str) -> list[Ref] | None: ...

    def contains(self, path: str) -> list[Ref]: ...

    def by_parent(self, parent_path: str) -> list[Ref]: ...

    def remove_file_subgraph(self, path: str) -> list[str]: ...

    # ------------------------------------------------------------------
    # Graph-level properties
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int: ...

    @property
    def edge_count(self) -> int: ...

    def is_dag(self) -> bool: ...

    # ------------------------------------------------------------------
    # Centrality algorithms
    # ------------------------------------------------------------------

    def pagerank(
        self,
        *,
        alpha: float = 0.85,
        personalization: dict[str, float] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> dict[str, float]: ...

    def betweenness_centrality(self, *, normalized: bool = True) -> dict[str, float]: ...

    def closeness_centrality(self) -> dict[str, float]: ...

    def katz_centrality(
        self,
        *,
        alpha: float = 0.1,
        beta: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> dict[str, float]: ...

    def degree_centrality(self) -> dict[str, float]: ...

    def in_degree_centrality(self) -> dict[str, float]: ...

    def out_degree_centrality(self) -> dict[str, float]: ...

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    def weakly_connected_components(self) -> list[set[str]]: ...

    def strongly_connected_components(self) -> list[set[str]]: ...

    def is_weakly_connected(self) -> bool: ...

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def ancestors(self, path: str) -> set[str]: ...

    def descendants(self, path: str) -> set[str]: ...

    def all_simple_paths(
        self, source: str, target: str, *, cutoff: int | None = None
    ) -> list[list[str]]: ...

    def topological_sort(self) -> list[str]: ...

    def shortest_path_length(self, source: str, target: str) -> float | None: ...

    # ------------------------------------------------------------------
    # Subgraph extraction
    # ------------------------------------------------------------------

    def subgraph(self, paths: list[str]) -> SubgraphResult: ...

    def neighborhood(
        self,
        path: str,
        *,
        max_depth: int = 2,
        direction: str = "both",
        edge_types: list[str] | None = None,
    ) -> SubgraphResult: ...

    def meeting_subgraph(self, start_paths: list[str], *, max_size: int = 50) -> SubgraphResult: ...

    def common_reachable(self, paths: list[str], *, direction: str = "forward") -> set[str]: ...

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def find_nodes(self, **attrs: object) -> list[str]: ...

    def find_edges(
        self,
        *,
        edge_type: str | None = None,
        source: str | None = None,
        target: str | None = None,
    ) -> list[tuple[str, str, dict[str, Any]]]: ...

    def edges_of(
        self,
        path: str,
        *,
        direction: str = "both",
        edge_types: list[str] | None = None,
    ) -> list[tuple[str, str, dict[str, Any]]]: ...

    # ------------------------------------------------------------------
    # Node similarity
    # ------------------------------------------------------------------

    def node_similarity(self, path1: str, path2: str, *, method: str = "jaccard") -> float: ...

    def similar_nodes(
        self, path: str, *, method: str = "jaccard", k: int = 10
    ) -> list[tuple[str, float]]: ...

    # ------------------------------------------------------------------
    # SQL persistence
    # ------------------------------------------------------------------

    async def to_sql(self, session: AsyncSession) -> None: ...

    async def from_sql(
        self, session: AsyncSession, file_model: type | None = None, *, path_prefix: str = ""
    ) -> None: ...


# Backward-compat aliases
GraphStore = GraphProvider
SupportsCentrality = GraphProvider
SupportsConnectivity = GraphProvider
SupportsTraversal = GraphProvider
SupportsSubgraph = GraphProvider
SupportsFiltering = GraphProvider
SupportsNodeSimilarity = GraphProvider
SupportsPersistence = GraphProvider
