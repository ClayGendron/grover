"""GraphMethodsMixin — graph delegates for DatabaseFileSystem."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grover.ref import Ref


class GraphMethodsMixin:
    """Delegates graph operations to ``self.graph_provider``.

    Methods are prefixed with ``graph_`` to avoid collisions with
    filesystem methods.  When ``self.graph_provider is None``, methods
    return appropriate empty/failure results.
    """

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def graph_add_node(self, path: str, **attrs: object) -> None:
        if self.graph_provider is None:  # type: ignore[attr-defined]
            return
        self.graph_provider.add_node(path, **attrs)  # type: ignore[attr-defined]

    def graph_remove_node(self, path: str) -> None:
        if self.graph_provider is None:  # type: ignore[attr-defined]
            return
        self.graph_provider.remove_node(path)  # type: ignore[attr-defined]

    def graph_has_node(self, path: str) -> bool:
        if self.graph_provider is None:  # type: ignore[attr-defined]
            return False
        return self.graph_provider.has_node(path)  # type: ignore[attr-defined]

    def graph_get_node(self, path: str) -> dict:
        if self.graph_provider is None:  # type: ignore[attr-defined]
            return {}
        return self.graph_provider.get_node(path)  # type: ignore[attr-defined]

    def graph_nodes(self) -> list[str]:
        if self.graph_provider is None:  # type: ignore[attr-defined]
            return []
        return self.graph_provider.nodes()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def graph_add_edge(self, source: str, target: str, edge_type: str, **attrs: object) -> None:
        if self.graph_provider is None:  # type: ignore[attr-defined]
            return
        self.graph_provider.add_edge(source, target, edge_type, **attrs)  # type: ignore[attr-defined]

    def graph_remove_edge(self, source: str, target: str) -> None:
        if self.graph_provider is None:  # type: ignore[attr-defined]
            return
        self.graph_provider.remove_edge(source, target)  # type: ignore[attr-defined]

    def graph_has_edge(self, source: str, target: str) -> bool:
        if self.graph_provider is None:  # type: ignore[attr-defined]
            return False
        return self.graph_provider.has_edge(source, target)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def graph_dependents(self, path: str) -> list[Ref]:
        if self.graph_provider is None:  # type: ignore[attr-defined]
            return []
        return self.graph_provider.dependents(path)  # type: ignore[attr-defined]

    def graph_dependencies(self, path: str) -> list[Ref]:
        if self.graph_provider is None:  # type: ignore[attr-defined]
            return []
        return self.graph_provider.dependencies(path)  # type: ignore[attr-defined]

    def graph_impacts(self, path: str, max_depth: int = 3) -> list[Ref]:
        if self.graph_provider is None:  # type: ignore[attr-defined]
            return []
        return self.graph_provider.impacts(path, max_depth)  # type: ignore[attr-defined]

    def graph_path_between(self, source: str, target: str) -> list[Ref] | None:
        if self.graph_provider is None:  # type: ignore[attr-defined]
            return None
        return self.graph_provider.path_between(source, target)  # type: ignore[attr-defined]

    def graph_contains(self, path: str) -> list[Ref]:
        if self.graph_provider is None:  # type: ignore[attr-defined]
            return []
        return self.graph_provider.contains(path)  # type: ignore[attr-defined]

    def graph_remove_file_subgraph(self, path: str) -> list[str]:
        if self.graph_provider is None:  # type: ignore[attr-defined]
            return []
        return self.graph_provider.remove_file_subgraph(path)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def graph_node_count(self) -> int:
        if self.graph_provider is None:  # type: ignore[attr-defined]
            return 0
        return self.graph_provider.node_count  # type: ignore[attr-defined]

    @property
    def graph_edge_count(self) -> int:
        if self.graph_provider is None:  # type: ignore[attr-defined]
            return 0
        return self.graph_provider.edge_count  # type: ignore[attr-defined]
