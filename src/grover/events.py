"""EventBus and event types for cross-layer consistency."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of filesystem events that trigger consistency updates."""

    FILE_WRITTEN = "file_written"
    FILE_DELETED = "file_deleted"
    FILE_MOVED = "file_moved"
    FILE_RESTORED = "file_restored"


@dataclass(frozen=True, slots=True)
class FileEvent:
    """Immutable record of a filesystem mutation.

    Attributes:
        event_type: The kind of mutation that occurred.
        path: Virtual path of the affected file (destination for moves).
        old_path: Previous path (moves only).
        content: File content when available (writes only), None otherwise.
    """

    event_type: EventType
    path: str
    old_path: str | None = None
    content: str | None = None
    user_id: str | None = None


class EventBus:
    """Dispatches filesystem events to registered handlers.

    Handlers are called sequentially in registration order.
    Exceptions are logged but never propagated — a failing handler
    degrades consistency, it does not crash the system.
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Callable[..., Any]]] = {et: [] for et in EventType}

    def register(self, event_type: EventType, handler: Callable[..., Any]) -> None:
        """Append *handler* to the list for *event_type*."""
        self._handlers[event_type].append(handler)

    def unregister(self, event_type: EventType, handler: Callable[..., Any]) -> bool:
        """Remove first occurrence of *handler*. Return True if found."""
        handlers = self._handlers[event_type]
        try:
            handlers.remove(handler)
            return True
        except ValueError:
            return False

    async def emit(self, event: FileEvent) -> None:
        """Dispatch *event* to all registered handlers for its type."""
        for handler in self._handlers[event.event_type]:
            try:
                await handler(event)
            except Exception:
                logger.warning(
                    "Handler %r failed for %s on %s",
                    handler,
                    event.event_type.value,
                    event.path,
                    exc_info=True,
                )

    @property
    def handler_count(self) -> int:
        """Total number of registered handlers across all event types."""
        return sum(len(h) for h in self._handlers.values())

    def clear(self) -> None:
        """Remove all registered handlers."""
        for handlers in self._handlers.values():
            handlers.clear()
