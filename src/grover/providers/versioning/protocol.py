"""Version provider protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from grover.results.operations import VerifyVersionResult


@runtime_checkable
class VersionProvider(Protocol):
    """Version storage — diff-based with periodic snapshots."""

    async def save_version(
        self,
        session: Any,
        file: Any,
        old_content: str,
        new_content: str,
        created_by: str = "agent",
    ) -> None: ...

    async def delete_versions(self, session: Any, file_id: str) -> None: ...

    async def list_versions(self, session: Any, file: Any) -> list: ...

    async def get_version_content(self, session: Any, file: Any, version: int) -> str | None: ...

    async def verify_chain(self, session: Any, file: Any) -> VerifyVersionResult: ...
