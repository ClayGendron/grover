"""GroverStore — LangGraph persistent store backed by Grover filesystem."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from grover.client import Grover, GroverAsync


class GroverStore(BaseStore):
    """LangGraph persistent store backed by Grover's versioned filesystem.

    Accepts either a sync :class:`~grover.Grover` or async
    :class:`~grover.GroverAsync` instance:

    - **Grover:** ``batch()`` works directly; ``abatch()`` raises ``TypeError``.
    - **GroverAsync:** ``abatch()`` calls native async API; ``batch()`` wraps
      via ``asyncio.run()``.

    Namespace tuples map to directory paths under a configurable prefix.
    Values are stored as JSON files.

    Usage::

        from grover import Grover
        from grover.integrations.langchain import GroverStore

        g = Grover()
        g.add_mount("data", backend)

        store = GroverStore(grover=g, prefix="/data/store")
        store.put(("users", "alice"), "prefs", {"theme": "dark"})
        item = store.get(("users", "alice"), "prefs")
        # item.value == {"theme": "dark"}

    Namespace ``("users", "alice", "notes")`` with key ``"idea-1"``
    maps to ``/data/store/users/alice/notes/idea-1.json``.
    """

    def __init__(
        self,
        grover: Grover | GroverAsync,
        *,
        prefix: str = "/store",
    ) -> None:
        from grover.client import GroverAsync

        self.grover = grover
        self.prefix = prefix.rstrip("/")
        self._is_async = isinstance(grover, GroverAsync)

    # ------------------------------------------------------------------
    # Abstract methods (required by BaseStore)
    # ------------------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        if self._is_async:
            return asyncio.run(self.abatch(list(ops)))

        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._handle_get(op))
            elif isinstance(op, PutOp):
                results.append(self._handle_put(op))
            elif isinstance(op, SearchOp):
                results.append(self._handle_search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._handle_list_namespaces(op))
            else:
                results.append(None)
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        if not self._is_async:
            raise TypeError(
                "Async methods require GroverAsync. Pass a GroverAsync instance or use sync methods instead."
            )

        results: list[Result] = []
        for op in list(ops):
            if isinstance(op, GetOp):
                results.append(await self._ahandle_get(op))
            elif isinstance(op, PutOp):
                results.append(await self._ahandle_put(op))
            elif isinstance(op, SearchOp):
                results.append(await self._ahandle_search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(await self._ahandle_list_namespaces(op))
            else:
                results.append(None)
        return results

    # ------------------------------------------------------------------
    # Sync operation handlers
    # ------------------------------------------------------------------

    def _handle_get(self, op: GetOp) -> Item | None:
        g = cast("Grover", self.grover)
        path = self._key_to_path(op.namespace, op.key)
        read_result = g.read(path)
        if not read_result.success or not read_result.file:
            return None
        if read_result.file.content is None:
            return None

        try:
            value = json.loads(read_result.file.content)
        except (json.JSONDecodeError, TypeError):
            return None

        now = datetime.now(tz=UTC)
        return Item(
            value=value,
            key=op.key,
            namespace=op.namespace,
            created_at=now,
            updated_at=now,
        )

    def _handle_put(self, op: PutOp) -> None:
        g = cast("Grover", self.grover)
        path = self._key_to_path(op.namespace, op.key)

        if op.value is None:
            # Delete
            g.delete(path, permanent=True)
        else:
            content = json.dumps(op.value, default=str)
            g.write(path, content, overwrite=True)

        return None

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        g = cast("Grover", self.grover)
        ns_dir = self._namespace_to_dir(op.namespace_prefix)
        now = datetime.now(tz=UTC)

        # Try semantic search if query is provided
        if op.query:
            try:
                result = g.vector_search(op.query, k=op.limit + op.offset)
            except Exception:
                result = None

            paths = [f.path for f in result.files] if result is not None and result.success else ()

            items: list[SearchItem] = []
            for path in paths:
                # Filter to only items under the namespace prefix
                if not path.startswith(ns_dir + "/"):
                    continue
                # Extract namespace and key from path
                ns, key = self._path_to_namespace_key(path)
                if ns is None:
                    continue

                # Read file content to get the stored value
                read_result = g.read(path)
                if not read_result.success or not read_result.file:
                    continue
                if read_result.file.content is None:
                    continue

                try:
                    value = json.loads(read_result.file.content)
                except (json.JSONDecodeError, TypeError):
                    value = {"content": read_result.file.content}

                items.append(
                    SearchItem(
                        namespace=ns,
                        key=key,
                        value=value,
                        created_at=now,
                        updated_at=now,
                    )
                )

            # Apply offset and limit
            return items[op.offset : op.offset + op.limit]

        # Fallback: list all items in namespace
        return self._list_items_in_namespace(op.namespace_prefix, limit=op.limit, offset=op.offset)

    def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        g = cast("Grover", self.grover)
        tree_result = g.tree(self.prefix)
        if not tree_result.success:
            return []

        namespaces = self._extract_namespaces(tree_result)
        return self._format_namespaces(namespaces, op)

    # ------------------------------------------------------------------
    # Async operation handlers
    # ------------------------------------------------------------------

    async def _ahandle_get(self, op: GetOp) -> Item | None:
        g = cast("GroverAsync", self.grover)
        path = self._key_to_path(op.namespace, op.key)
        read_result = await g.read(path)
        if not read_result.success or not read_result.file:
            return None
        if read_result.file.content is None:
            return None

        try:
            value = json.loads(read_result.file.content)
        except (json.JSONDecodeError, TypeError):
            return None

        now = datetime.now(tz=UTC)
        return Item(
            value=value,
            key=op.key,
            namespace=op.namespace,
            created_at=now,
            updated_at=now,
        )

    async def _ahandle_put(self, op: PutOp) -> None:
        g = cast("GroverAsync", self.grover)
        path = self._key_to_path(op.namespace, op.key)

        if op.value is None:
            await g.delete(path, permanent=True)
        else:
            content = json.dumps(op.value, default=str)
            await g.write(path, content, overwrite=True)

        return None

    async def _ahandle_search(self, op: SearchOp) -> list[SearchItem]:
        g = cast("GroverAsync", self.grover)
        ns_dir = self._namespace_to_dir(op.namespace_prefix)
        now = datetime.now(tz=UTC)

        if op.query:
            try:
                result = await g.vector_search(op.query, k=op.limit + op.offset)
            except Exception:
                result = None

            paths = [f.path for f in result.files] if result is not None and result.success else ()

            items: list[SearchItem] = []
            for path in paths:
                if not path.startswith(ns_dir + "/"):
                    continue
                ns, key = self._path_to_namespace_key(path)
                if ns is None:
                    continue

                read_result = await g.read(path)
                if not read_result.success or not read_result.file:
                    continue
                if read_result.file.content is None:
                    continue

                try:
                    value = json.loads(read_result.file.content)
                except (json.JSONDecodeError, TypeError):
                    value = {"content": read_result.file.content}

                items.append(
                    SearchItem(
                        namespace=ns,
                        key=key,
                        value=value,
                        created_at=now,
                        updated_at=now,
                    )
                )

            return items[op.offset : op.offset + op.limit]

        return await self._alist_items_in_namespace(op.namespace_prefix, limit=op.limit, offset=op.offset)

    async def _ahandle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        g = cast("GroverAsync", self.grover)
        tree_result = await g.tree(self.prefix)
        if not tree_result.success:
            return []

        namespaces = self._extract_namespaces(tree_result)
        return self._format_namespaces(namespaces, op)

    # ------------------------------------------------------------------
    # Namespace / path mapping
    # ------------------------------------------------------------------

    def _namespace_to_dir(self, namespace: tuple[str, ...]) -> str:
        """Convert a namespace tuple to a directory path."""
        if not namespace:
            return self.prefix
        return self.prefix + "/" + "/".join(namespace)

    def _key_to_path(self, namespace: tuple[str, ...], key: str) -> str:
        """Convert namespace + key to a file path."""
        ns_dir = self._namespace_to_dir(namespace)
        return f"{ns_dir}/{key}.json"

    def _path_to_namespace_key(self, path: str) -> tuple[tuple[str, ...] | None, str]:
        """Extract namespace and key from a file path."""
        if not path.startswith(self.prefix + "/"):
            return None, ""

        relative = path[len(self.prefix) + 1 :]
        parts = relative.split("/")
        if len(parts) < 2:
            return None, ""

        key = parts[-1]
        if key.endswith(".json"):
            key = key[:-5]

        namespace = tuple(parts[:-1])
        return namespace, key

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_namespaces(self, tree_result: object) -> set[tuple[str, ...]]:
        """Extract unique namespace tuples from a tree result."""
        from grover.models.internal.evidence import TreeEvidence

        namespaces: set[tuple[str, ...]] = set()
        prefix_len = len(self.prefix) + 1  # +1 for trailing /

        for f in tree_result.files:  # type: ignore[union-attr]
            is_dir = any(isinstance(e, TreeEvidence) and e.is_directory for e in f.evidence)
            if is_dir:
                continue
            if not f.path.startswith(self.prefix + "/"):
                continue

            relative = f.path[prefix_len:]
            parts = relative.split("/")
            if len(parts) < 2:
                continue

            ns_parts = tuple(parts[:-1])
            namespaces.add(ns_parts)

        return namespaces

    @staticmethod
    def _format_namespaces(
        namespaces: set[tuple[str, ...]],
        op: ListNamespacesOp,
    ) -> list[tuple[str, ...]]:
        """Apply match conditions, max_depth, offset, and limit to namespaces."""
        filtered = list(namespaces)
        if op.match_conditions:
            filtered = GroverStore._apply_match_conditions(filtered, op.match_conditions)

        if op.max_depth is not None:
            filtered = sorted({ns[: op.max_depth] for ns in filtered})
        else:
            filtered.sort()

        return filtered[op.offset : op.offset + op.limit]

    def _list_items_in_namespace(
        self,
        namespace: tuple[str, ...],
        *,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SearchItem]:
        """List all items in a namespace (used as fallback for search)."""
        g = cast("Grover", self.grover)
        ns_dir = self._namespace_to_dir(namespace)
        now = datetime.now(tz=UTC)

        tree_result = g.tree(ns_dir)
        if not tree_result.success:
            return []

        from grover.models.internal.evidence import TreeEvidence

        items: list[SearchItem] = []
        for f in tree_result.files:
            is_dir = any(isinstance(e, TreeEvidence) and e.is_directory for e in f.evidence)
            if is_dir:
                continue
            if not f.path.endswith(".json"):
                continue

            ns, key = self._path_to_namespace_key(f.path)
            if ns is None:
                continue

            read_result = g.read(f.path)
            if not read_result.success or not read_result.file:
                continue
            if read_result.file.content is None:
                continue

            try:
                value = json.loads(read_result.file.content)
            except (json.JSONDecodeError, TypeError):
                continue

            items.append(
                SearchItem(
                    namespace=ns,
                    key=key,
                    value=value,
                    created_at=now,
                    updated_at=now,
                )
            )

        return items[offset : offset + limit]

    async def _alist_items_in_namespace(
        self,
        namespace: tuple[str, ...],
        *,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SearchItem]:
        """Async variant of _list_items_in_namespace."""
        g = cast("GroverAsync", self.grover)
        ns_dir = self._namespace_to_dir(namespace)
        now = datetime.now(tz=UTC)

        tree_result = await g.tree(ns_dir)
        if not tree_result.success:
            return []

        from grover.models.internal.evidence import TreeEvidence

        items: list[SearchItem] = []
        for f in tree_result.files:
            is_dir = any(isinstance(e, TreeEvidence) and e.is_directory for e in f.evidence)
            if is_dir:
                continue
            if not f.path.endswith(".json"):
                continue

            ns, key = self._path_to_namespace_key(f.path)
            if ns is None:
                continue

            read_result = await g.read(f.path)
            if not read_result.success or not read_result.file:
                continue
            if read_result.file.content is None:
                continue

            try:
                value = json.loads(read_result.file.content)
            except (json.JSONDecodeError, TypeError):
                continue

            items.append(
                SearchItem(
                    namespace=ns,
                    key=key,
                    value=value,
                    created_at=now,
                    updated_at=now,
                )
            )

        return items[offset : offset + limit]

    @staticmethod
    def _apply_match_conditions(
        namespaces: list[tuple[str, ...]],
        conditions: tuple[MatchCondition, ...],
    ) -> list[tuple[str, ...]]:
        """Filter namespaces based on match conditions."""
        result = namespaces
        for cond in conditions:
            match_type = cond.match_type
            pattern = cond.path

            if match_type == "prefix":
                result = [
                    ns
                    for ns in result
                    if len(ns) >= len(pattern) and all(p == "*" or p == n for p, n in zip(pattern, ns, strict=False))
                ]
            elif match_type == "suffix":
                result = [
                    ns
                    for ns in result
                    if len(ns) >= len(pattern)
                    and all(p == "*" or p == n for p, n in zip(reversed(pattern), reversed(ns), strict=False))
                ]

        return result
