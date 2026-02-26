"""Mount — first-class composition unit for filesystem, graph, and search."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grover.fs.permissions import Permission
from grover.fs.utils import normalize_path

from .errors import ProtocolConflictError, ProtocolNotAvailableError
from .protocols import DISPATCH_PROTOCOLS

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from grover.fs.protocol import StorageBackend
    from grover.graph.protocols import GraphStore
    from grover.search._engine import SearchEngine


class Mount:
    """A mount point composing filesystem, graph, and search components.

    Each mount has:
    - ``filesystem`` — the storage backend (required)
    - ``graph`` — an optional in-memory graph for this mount
    - ``search`` — an optional search engine for this mount

    Protocol dispatch checks all three components at construction time and
    builds a dispatch map.  If two components implement the same dispatch
    protocol, ``ProtocolConflictError`` is raised.
    """

    def __init__(
        self,
        path: str = "",
        filesystem: StorageBackend | None = None,
        *,
        graph: GraphStore | None = None,
        search: SearchEngine | None = None,
        session_factory: Callable[..., AsyncSession] | None = None,
        permission: Permission = Permission.READ_WRITE,
        label: str = "",
        mount_type: str = "vfs",
        hidden: bool = False,
        read_only_paths: set[str] | None = None,
    ) -> None:
        self.path: str = normalize_path(path).rstrip("/")
        self.filesystem: StorageBackend | None = filesystem
        self.graph: GraphStore | None = graph
        self.search: SearchEngine | None = search
        self.session_factory: Callable[..., AsyncSession] | None = session_factory
        self.permission: Permission = permission
        self.label: str = label or self.path.lstrip("/") or "root"
        self.mount_type: str = mount_type
        self.hidden: bool = hidden
        self.read_only_paths: set[str] = read_only_paths if read_only_paths is not None else set()
        self._dispatch_map: dict[type, tuple[str, object]] = self._build_dispatch_map()

    # ------------------------------------------------------------------
    # Protocol dispatch
    # ------------------------------------------------------------------

    def dispatch(self, protocol: type) -> object:
        """Return the component implementing *protocol*.

        Raises
        ------
        ProtocolNotAvailableError
            If no component implements the requested protocol.
        """
        entry = self._dispatch_map.get(protocol)
        if entry is None:
            raise ProtocolNotAvailableError(
                f"{protocol.__name__} not available. Check mount configuration at '{self.path}'."
            )
        return entry[1]

    def has_capability(self, protocol: type) -> bool:
        """Check if any component implements *protocol*."""
        return protocol in self._dispatch_map

    def supported_protocols(self) -> set[type]:
        """Return all dispatch protocols available on this mount."""
        return set(self._dispatch_map.keys())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_dispatch_map(self) -> dict[type, tuple[str, object]]:
        """Build the protocol → component dispatch map.

        Checks each component against dispatch protocols.  Components that
        expose a ``supported_protocols()`` method (like ``SearchEngine``)
        use that.  Others are checked via ``isinstance()``.
        """
        dmap: dict[type, tuple[str, object]] = {}
        components: list[tuple[str, object]] = [
            ("filesystem", self.filesystem),
            ("graph", self.graph),
            ("search", self.search),
        ]
        for name, comp in components:
            if comp is None:
                continue
            # Get protocols this component satisfies
            if hasattr(comp, "supported_protocols") and callable(comp.supported_protocols):
                protos: list[type] = comp.supported_protocols()  # type: ignore[assignment]
            else:
                protos = [p for p in DISPATCH_PROTOCOLS if isinstance(comp, p)]
            for proto in protos:
                if proto in dmap:
                    existing_name = dmap[proto][0]
                    raise ProtocolConflictError(
                        f"{proto.__name__} implemented by both '{existing_name}' and '{name}'"
                    )
                dmap[proto] = (name, comp)
        return dmap

    def __repr__(self) -> str:
        parts = [f"path={self.path!r}"]
        if self.filesystem is not None:
            parts.append(f"filesystem={type(self.filesystem).__name__}")
        if self.graph is not None:
            parts.append(f"graph={type(self.graph).__name__}")
        if self.search is not None:
            parts.append(f"search={type(self.search).__name__}")
        return f"Mount({', '.join(parts)})"
