"""GroverFileSystem — the protocol every backend must implement.

This is the narrow waist of Grover: a small set of operations that work on
any path regardless of entity kind.  The facade routes calls to the correct
backend by mount prefix; the backend implements these methods against its
storage layer (database, local disk, external API, etc.).

All methods return ``GroverResult`` without a back-reference — the facade
stamps ``_grover`` after receiving the result.

Chainable CRUD methods accept both ``path`` (single) and ``candidates``
(batch).  The backend resolves a paths list once at the top and runs one
query regardless of count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.results import GroverResult


@runtime_checkable
class GroverFileSystem(Protocol):
    """The single protocol every Grover backend implements."""

    # -------------------------------------------------------------------
    # CRUD — chainable
    # -------------------------------------------------------------------

    async def read(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def stat(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def edit(
        self,
        path: str | None = None,
        old: str = "",
        new: str = "",
        candidates: GroverResult | None = None,
        replace_all: bool = False,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def ls(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def delete(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        permanent: bool = False,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    # -------------------------------------------------------------------
    # CRUD — path-only
    # -------------------------------------------------------------------

    async def write(
        self,
        path: str,
        content: str,
        overwrite: bool = True,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def move(
        self,
        src: str,
        dest: str,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def copy(
        self,
        src: str,
        dest: str,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def mkdir(
        self,
        path: str,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def mkconn(
        self,
        source: str,
        target: str,
        connection_type: str,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def tree(
        self,
        path: str,
        max_depth: int | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    # -------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------

    async def glob(
        self,
        pattern: str,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def grep(
        self,
        pattern: str,
        case_sensitive: bool = True,
        max_results: int | None = None,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def semantic_search(
        self,
        query: str,
        k: int = 15,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def vector_search(
        self,
        vector: list[float],
        k: int = 15,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def lexical_search(
        self,
        query: str,
        k: int = 15,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    # -------------------------------------------------------------------
    # Graph — require candidates as input
    # -------------------------------------------------------------------

    async def predecessors(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def successors(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def ancestors(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def descendants(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def neighborhood(
        self,
        path: str | None = None,
        candidates: GroverResult | None = None,
        *,
        depth: int = 2,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def meeting_subgraph(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def min_meeting_subgraph(
        self,
        candidates: GroverResult,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def pagerank(
        self,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def betweenness_centrality(
        self,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def closeness_centrality(
        self,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def degree_centrality(
        self,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def in_degree_centrality(
        self,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def out_degree_centrality(
        self,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...

    async def hits(
        self,
        candidates: GroverResult | None = None,
        *,
        session: AsyncSession | None = None,
    ) -> GroverResult: ...
