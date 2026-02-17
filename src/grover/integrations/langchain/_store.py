"""GroverStore — LangGraph persistent store backed by Grover filesystem."""

import asyncio
import json
from collections.abc import Iterable
from datetime import UTC, datetime

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

from grover._grover import Grover


class GroverStore(BaseStore):
    """LangGraph persistent store backed by Grover's versioned filesystem.

    Namespace tuples map to directory paths under a configurable prefix.
    Values are stored as JSON files.

    Usage::

        from grover import Grover
        from grover.integrations.langchain import GroverStore

        g = Grover()
        g.mount("/data", backend)

        store = GroverStore(grover=g, prefix="/data/store")
        store.put(("users", "alice"), "prefs", {"theme": "dark"})
        item = store.get(("users", "alice"), "prefs")
        # item.value == {"theme": "dark"}

    Namespace ``("users", "alice", "notes")`` with key ``"idea-1"``
    maps to ``/data/store/users/alice/notes/idea-1.json``.
    """

    def __init__(
        self,
        grover: Grover,
        *,
        prefix: str = "/store",
    ) -> None:
        self.grover = grover
        self.prefix = prefix.rstrip("/")

    # ------------------------------------------------------------------
    # Abstract methods (required by BaseStore)
    # ------------------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
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
        return await asyncio.to_thread(self.batch, list(ops))

    # ------------------------------------------------------------------
    # Operation handlers
    # ------------------------------------------------------------------

    def _handle_get(self, op: GetOp) -> Item | None:
        path = self._key_to_path(op.namespace, op.key)
        read_result = self.grover.read(path)
        if not read_result.success or read_result.content is None:
            return None

        try:
            value = json.loads(read_result.content)
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
        path = self._key_to_path(op.namespace, op.key)

        if op.value is None:
            # Delete
            self.grover.delete(path, permanent=True)
        else:
            content = json.dumps(op.value, default=str)
            self.grover.write(path, content, overwrite=True)

        return None

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        ns_dir = self._namespace_to_dir(op.namespace_prefix)
        now = datetime.now(tz=UTC)

        # Try semantic search if query is provided
        if op.query:
            try:
                results = self.grover.search(op.query, k=op.limit + op.offset)
            except Exception:
                results = []

            items: list[SearchItem] = []
            for sr in results:
                path = sr.ref.path
                # Filter to only items under the namespace prefix
                if not path.startswith(ns_dir + "/"):
                    continue
                # Extract namespace and key from path
                ns, key = self._path_to_namespace_key(path)
                if ns is None:
                    continue

                try:
                    value = json.loads(sr.content)
                except (json.JSONDecodeError, TypeError):
                    value = {"content": sr.content}

                items.append(
                    SearchItem(
                        namespace=ns,
                        key=key,
                        value=value,
                        created_at=now,
                        updated_at=now,
                        score=sr.score,
                    )
                )

            # Apply offset and limit
            return items[op.offset : op.offset + op.limit]

        # Fallback: list all items in namespace
        return self._list_items_in_namespace(
            op.namespace_prefix, limit=op.limit, offset=op.offset
        )

    def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        tree_result = self.grover.tree(self.prefix)
        if not tree_result.success:
            return []

        # Collect unique namespace tuples from all file paths
        namespaces: set[tuple[str, ...]] = set()
        prefix_len = len(self.prefix) + 1  # +1 for trailing /

        for entry in tree_result.entries:
            if entry.is_directory:
                continue
            path = entry.path
            if not path.startswith(self.prefix + "/"):
                continue

            # Extract relative path and strip the key filename
            relative = path[prefix_len:]
            parts = relative.split("/")
            if len(parts) < 2:
                continue

            # namespace = all parts except the last (which is the key file)
            ns_parts = tuple(parts[:-1])
            namespaces.add(ns_parts)

        # Apply match conditions
        filtered = list(namespaces)
        if op.match_conditions:
            filtered = self._apply_match_conditions(filtered, op.match_conditions)

        # Apply max_depth (truncate and deduplicate per LangGraph spec)
        if op.max_depth is not None:
            filtered = sorted({ns[:op.max_depth] for ns in filtered})
        else:
            filtered.sort()

        # Apply offset and limit
        return filtered[op.offset : op.offset + op.limit]

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

    def _list_items_in_namespace(
        self,
        namespace: tuple[str, ...],
        *,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SearchItem]:
        """List all items in a namespace (used as fallback for search)."""
        ns_dir = self._namespace_to_dir(namespace)
        now = datetime.now(tz=UTC)

        tree_result = self.grover.tree(ns_dir)
        if not tree_result.success:
            return []

        items: list[SearchItem] = []
        for entry in tree_result.entries:
            if entry.is_directory:
                continue
            if not entry.path.endswith(".json"):
                continue

            ns, key = self._path_to_namespace_key(entry.path)
            if ns is None:
                continue

            read_result = self.grover.read(entry.path)
            if not read_result.success or read_result.content is None:
                continue

            try:
                value = json.loads(read_result.content)
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
                    if len(ns) >= len(pattern)
                    and all(
                        p == "*" or p == n for p, n in zip(pattern, ns, strict=False)
                    )
                ]
            elif match_type == "suffix":
                result = [
                    ns
                    for ns in result
                    if len(ns) >= len(pattern)
                    and all(
                        p == "*" or p == n
                        for p, n in zip(
                            reversed(pattern), reversed(ns), strict=False
                        )
                    )
                ]

        return result
