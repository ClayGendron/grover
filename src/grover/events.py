"""EventBus and event types for cross-layer consistency."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of filesystem events that trigger consistency updates."""

    FILE_WRITTEN = "file_written"
    FILE_DELETED = "file_deleted"
    FILE_MOVED = "file_moved"
    FILE_RESTORED = "file_restored"
    CONNECTION_ADDED = "connection_added"
    CONNECTION_DELETED = "connection_deleted"


class IndexingMode(Enum):
    """Controls how file events are dispatched to indexing handlers.

    ``BACKGROUND`` (default): events are debounced per-path and dispatched
    in background ``asyncio.Task`` instances so that ``write()`` / ``edit()``
    return immediately.

    ``MANUAL``: all event dispatch is suppressed.  Only an explicit call to
    ``index()`` populates the graph and search engine.
    """

    BACKGROUND = "background"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class FileEvent:
    """Immutable record of a filesystem mutation.

    Attributes:
        event_type: The kind of mutation that occurred.
        path: Virtual path of the affected entity. For files this is the
            file path; for connections it is ``source[type]target``.
        old_path: Previous path (moves only).
        content: File content when available (writes only), None otherwise.
        user_id: User who triggered the event.
        source_path: Source file (CONNECTION_ADDED / CONNECTION_DELETED).
        target_path: Target file (CONNECTION_ADDED / CONNECTION_DELETED).
        connection_type: Edge type string (CONNECTION_ADDED / CONNECTION_DELETED).
        weight: Edge weight (CONNECTION_ADDED, default 1.0).
    """

    event_type: EventType
    path: str
    old_path: str | None = None
    content: str | None = None
    user_id: str | None = None
    # Connection context (CONNECTION_ADDED / CONNECTION_DELETED)
    source_path: str | None = None
    target_path: str | None = None
    connection_type: str | None = None
    weight: float = 1.0


class EventBus:
    """Dispatches filesystem events to registered handlers.

    In ``BACKGROUND`` mode (default), file mutation events (WRITTEN, DELETED,
    MOVED, RESTORED) are debounced per-path and processed in background
    ``asyncio.Task`` instances.  Connection events are dispatched immediately
    as background tasks (no debounce — they are lightweight).

    In ``MANUAL`` mode, ``emit()`` is a no-op; the caller is responsible for
    calling ``index()`` explicitly.

    Handlers are called sequentially in registration order within a single
    dispatch.  Exceptions are logged but never propagated — a failing handler
    degrades consistency, it does not crash the system.
    """

    def __init__(
        self,
        *,
        indexing_mode: IndexingMode = IndexingMode.BACKGROUND,
        debounce_delay: float = 0.1,
    ) -> None:
        self._handlers: dict[EventType, list[Callable[..., Any]]] = {et: [] for et in EventType}
        self._indexing_mode = indexing_mode
        self._debounce_delay = debounce_delay
        # Per-path pending events: path -> (latest_event, TimerHandle | None)
        self._pending: dict[str, tuple[FileEvent, asyncio.TimerHandle | None]] = {}
        # Currently-running background tasks
        self._active_tasks: set[asyncio.Task[None]] = set()
        # True when we're inside _dispatch (to handle nested emit from handlers)
        self._dispatching: bool = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    async def emit(self, event: FileEvent) -> None:
        """Dispatch *event* according to the current indexing mode.

        In ``MANUAL`` mode this is a no-op.  In ``BACKGROUND`` mode the
        event is scheduled for background dispatch (debounced for writes
        and restores).  If called from within a running handler (nested
        emit), the event is dispatched inline to preserve ordering.
        """
        if self._indexing_mode == IndexingMode.MANUAL:
            return

        # Nested emit from within a handler (e.g. CONNECTION_ADDED from
        # _analyze_and_integrate) — dispatch inline to preserve ordering
        if self._dispatching:
            await self._dispatch(event)
            return

        self._schedule_background(event)

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, event: FileEvent) -> None:
        """Sequentially call all handlers for *event* (inline)."""
        prev = self._dispatching
        self._dispatching = True
        try:
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
        finally:
            self._dispatching = prev

    def _schedule_background(self, event: FileEvent) -> None:
        """Queue *event* for background dispatch with per-path debouncing."""
        loop = asyncio.get_running_loop()
        et = event.event_type

        if et in (EventType.FILE_WRITTEN, EventType.FILE_RESTORED):
            # Debounce by path — cancel existing timer, store latest event
            path = event.path
            if path in self._pending:
                _, old_handle = self._pending[path]
                if old_handle is not None:
                    old_handle.cancel()
            handle = loop.call_later(self._debounce_delay, self._fire_background, path)
            self._pending[path] = (event, handle)

        elif et == EventType.FILE_DELETED:
            # Cancel any pending WRITTEN for this path (file is gone)
            path = event.path
            if path in self._pending:
                _, old_handle = self._pending.pop(path)
                if old_handle is not None:
                    old_handle.cancel()
            # Fire delete immediately
            self._create_background_task(self._dispatch(event))

        elif et == EventType.FILE_MOVED:
            # Cancel any pending event for old path
            if event.old_path and event.old_path in self._pending:
                _, old_handle = self._pending.pop(event.old_path)
                if old_handle is not None:
                    old_handle.cancel()
            # Fire move immediately
            self._create_background_task(self._dispatch(event))

        else:
            # CONNECTION_ADDED / CONNECTION_DELETED — fire immediately
            self._create_background_task(self._dispatch(event))

    def _fire_background(self, path: str) -> None:
        """Timer callback: pop the pending event for *path* and dispatch it."""
        entry = self._pending.pop(path, None)
        if entry is not None:
            event, _ = entry
            self._create_background_task(self._dispatch(event))

    def _create_background_task(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        """Create a tracked background task from *coro*."""
        task = asyncio.get_running_loop().create_task(coro)
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)
        return task

    # ------------------------------------------------------------------
    # Drain / lifecycle
    # ------------------------------------------------------------------

    async def drain(self) -> None:
        """Fire all pending timers immediately, then await all active tasks.

        Loops until settled (handlers may emit new events during drain).
        """
        while self._pending or self._active_tasks:
            # Fire all pending timers
            for path in list(self._pending):
                event, handle = self._pending.pop(path)
                if handle is not None:
                    handle.cancel()
                self._create_background_task(self._dispatch(event))
            # Await all active tasks
            if self._active_tasks:
                await asyncio.gather(*list(self._active_tasks), return_exceptions=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def indexing_mode(self) -> IndexingMode:
        """The current indexing mode."""
        return self._indexing_mode

    @property
    def pending_count(self) -> int:
        """Number of pending (debounced) + active background events."""
        return len(self._pending) + len(self._active_tasks)

    @property
    def handler_count(self) -> int:
        """Total number of registered handlers across all event types."""
        return sum(len(h) for h in self._handlers.values())

    def clear(self) -> None:
        """Remove all registered handlers."""
        for handlers in self._handlers.values():
            handlers.clear()
