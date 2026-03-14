"""Graph protocol — unified runtime-checkable interface for graph backends.

All graph capabilities (centrality, traversal, subgraph extraction, filtering,
persistence) are part of a single ``GraphProvider`` protocol.
``RustworkxGraph`` is the sole implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.models.internal.results import FileSearchResult, FileSearchSet


@runtime_checkable
class GraphProvider(Protocol):
    """Graph interface — nodes are file paths, edges are dependencies.

    Mutations are synchronous. Query/algorithm methods are async.
    All async query methods require a ``session`` parameter so the
    provider can self-refresh from the database when stale.

    Query methods accept ``FileSearchSet`` as their first positional
    argument for composability — the output of one graph operation
    can feed directly into another via set algebra.
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
    def graph(self) -> Any: ...

    # ------------------------------------------------------------------
    # Graph APIs — async typed result returns
    # ------------------------------------------------------------------

    async def predecessors(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        """One-hop backward: all nodes with an edge pointing *to* any node
        in *candidates*, excluding the candidate nodes themselves.

        For multi-path input the result is the union across all input paths.
        """
        ...

    async def successors(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        """One-hop forward: all nodes that any node in *candidates* points
        *to*, excluding the candidate nodes themselves.

        For multi-path input the result is the union across all input paths.
        """
        ...

    async def ancestors(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        """Transitive backward: all nodes reachable by following edges
        backward from any node in *candidates*, recursively until no more
        predecessors exist. Excludes the candidate nodes themselves.

        For multi-path input the result is the union of all ancestor sets.
        """
        ...

    async def descendants(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        """Transitive forward: all nodes reachable by following edges
        forward from any node in *candidates*, recursively until no more
        successors exist. Excludes the candidate nodes themselves.

        For multi-path input the result is the union of all descendant sets.
        """
        ...

    async def neighborhood(
        self,
        candidates: FileSearchSet,
        *,
        max_depth: int = 2,
        session: AsyncSession,
    ) -> FileSearchResult:
        """Bounded undirected BFS around a single node up to *max_depth* hops.

        Returns all discovered nodes and the edges between them (induced
        subgraph of the visited set). Follows edges in both directions.

        Requires exactly 1 path in *candidates* (``ValueError`` otherwise).
        """
        ...

    async def meeting_subgraph(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        """Find all paths between candidate nodes.

        For every ordered pair (A, B), finds all directed paths from A
        to B following edge direction. Since every pair is checked in
        both orders, reverse paths are covered naturally.

        If candidates span disconnected subgraphs, bridges them by
        finding nearest common descendants first, then nearest common
        ancestors if still disconnected.

        Returns the candidate nodes, all intermediate nodes on those
        paths, and the edges between them.

        Result includes both ``files`` (nodes) and ``connections`` (edges).
        """
        ...

    async def min_meeting_subgraph(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        """Minimum connecting subgraph between candidate nodes.

        Starts from ``meeting_subgraph`` then iteratively removes non-candidate
        nodes that are not articulation points (whose removal would not
        disconnect the graph).

        Result includes both ``files`` (nodes) and ``connections`` (edges).
        """
        ...

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
        """PageRank importance scores for candidate nodes.

        *alpha* is the damping factor. *personalization* biases the
        random walk toward specific nodes (path → weight mapping).
        """
        ...

    async def betweenness_centrality(
        self,
        candidates: FileSearchSet,
        *,
        normalized: bool = True,
        session: AsyncSession,
    ) -> FileSearchResult:
        """Betweenness centrality — how often each candidate lies on
        shortest paths between other nodes in the topology.
        """
        ...

    async def closeness_centrality(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        """Closeness centrality — inverse of the average shortest-path
        distance from each candidate to all other reachable nodes.
        """
        ...

    async def hits(
        self,
        candidates: FileSearchSet,
        *,
        max_iter: int = 100,
        tol: float = 1e-8,
        session: AsyncSession,
    ) -> FileSearchResult:
        """HITS hub and authority scores. Each candidate gets two
        evidence entries: ``hits_hub`` and ``hits_authority``.
        """
        ...

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
        """Katz centrality — measures influence by counting all paths
        from a node, with longer paths attenuated by *alpha*.
        """
        ...

    async def degree_centrality(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        """Degree centrality — fraction of nodes each candidate is
        connected to (in + out edges combined).
        """
        ...

    async def in_degree_centrality(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        """In-degree centrality — fraction of nodes with an edge
        pointing *to* each candidate. Measures how depended-upon
        a file is.
        """
        ...

    async def out_degree_centrality(self, candidates: FileSearchSet, *, session: AsyncSession) -> FileSearchResult:
        """Out-degree centrality — fraction of nodes each candidate
        points *to*. Measures how many dependencies a file has.
        """
        ...
