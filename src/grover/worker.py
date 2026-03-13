"""BackgroundWorker — debounced background task scheduling."""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

logger = logging.getLogger(__name__)


class IndexingMode(Enum):
    """Controls how file events are dispatched to indexing handlers.

    ``BACKGROUND`` (default): work is debounced per-key and dispatched
    in background ``asyncio.Task`` instances so that ``write()`` / ``edit()``
    return immediately.

    ``MANUAL``: all scheduling is suppressed.  Only an explicit call to
    ``index()`` populates the graph and search engine.
    """

    BACKGROUND = "background"
    MANUAL = "manual"


class BackgroundWorker:
    """Debounced background task scheduler.

    In ``BACKGROUND`` mode (default), ``schedule()`` debounces work per-key
    and runs it in background ``asyncio.Task`` instances.
    ``schedule_immediate()`` creates tasks without debouncing.

    In ``MANUAL`` mode, both methods are no-ops.

    Exceptions in tasks are logged but never propagated — a failing task
    degrades consistency, it does not crash the system.
    """

    _MAX_DRAIN_ITERATIONS = 50

    def __init__(
        self,
        *,
        indexing_mode: IndexingMode = IndexingMode.BACKGROUND,
        debounce_delay: float = 0.1,
        drain_timeout: float = 30.0,
    ) -> None:
        self._indexing_mode = indexing_mode
        self._debounce_delay = debounce_delay
        self._drain_timeout = drain_timeout
        # Per-key pending work: key -> (coro_factory, TimerHandle | None)
        self._pending: dict[
            str,
            tuple[Callable[[], Coroutine[object, object, None]], asyncio.TimerHandle | None],
        ] = {}
        # Currently-running background tasks
        self._active_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def schedule(
        self, key: str, coro_factory: Callable[[], Coroutine[object, object, None]]
    ) -> None:
        """Schedule debounced work for *key*.

        If there is already pending work for *key*, it is replaced.
        The *coro_factory* is called (creating the coroutine) only when
        the debounce timer fires.  No-op in ``MANUAL`` mode.
        """
        if self._indexing_mode == IndexingMode.MANUAL:
            return

        loop = asyncio.get_running_loop()

        # Cancel existing timer for this key
        if key in self._pending:
            _, old_handle = self._pending[key]
            if old_handle is not None:
                old_handle.cancel()

        handle = loop.call_later(self._debounce_delay, self._fire_pending, key)
        self._pending[key] = (coro_factory, handle)

    def schedule_immediate(self, coro: Coroutine[object, object, None]) -> None:
        """Run *coro* as a background task immediately (no debounce).

        No-op in ``MANUAL`` mode.  If suppressed, the coroutine is closed
        to prevent 'coroutine was never awaited' warnings.
        """
        if self._indexing_mode == IndexingMode.MANUAL:
            coro.close()
            return

        self._create_task(coro)

    def cancel(self, key: str) -> None:
        """Cancel any pending work for *key*.

        Removes the key from the pending dict and cancels its timer.
        Does nothing if no work is pending for *key*.
        """
        entry = self._pending.pop(key, None)
        if entry is not None:
            _, handle = entry
            if handle is not None:
                handle.cancel()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fire_pending(self, key: str) -> None:
        """Timer callback: pop the pending factory for *key* and run it."""
        entry = self._pending.pop(key, None)
        if entry is not None:
            factory, _ = entry
            try:
                self._create_task(factory())
            except Exception:
                logger.warning("Factory for key %r failed", key, exc_info=True)

    def _create_task(self, coro: Coroutine[object, object, None]) -> asyncio.Task[None]:
        """Create a tracked background task from *coro* with error isolation."""
        task = asyncio.get_running_loop().create_task(coro)
        self._active_tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[None]) -> None:
        """Done callback: discard task and log any exception."""
        self._active_tasks.discard(task)
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                logger.warning("Background task failed: %s", exc, exc_info=exc)

    # ------------------------------------------------------------------
    # Drain / lifecycle
    # ------------------------------------------------------------------

    async def drain(self, *, timeout: float | None = None) -> None:
        """Fire all pending timers immediately, then await all active tasks.

        Loops until settled (tasks may schedule new work during drain).
        Times out after *timeout* seconds (defaults to ``drain_timeout``
        from the constructor) and cancels remaining tasks.
        """
        effective = timeout if timeout is not None else self._drain_timeout
        loop = asyncio.get_running_loop()
        deadline = loop.time() + effective
        iterations = 0

        while self._pending or self._active_tasks:
            iterations += 1
            if iterations > self._MAX_DRAIN_ITERATIONS:
                logger.warning(
                    "drain: exceeded %d iterations, cancelling %d tasks",
                    self._MAX_DRAIN_ITERATIONS,
                    len(self._active_tasks),
                )
                await self._cancel_active()
                self._pending.clear()
                return

            # Fire all pending timers
            for key in list(self._pending):
                factory, handle = self._pending.pop(key)
                if handle is not None:
                    handle.cancel()
                try:
                    self._create_task(factory())
                except Exception:
                    logger.warning("Factory for key %r failed during drain", key, exc_info=True)

            # Await active tasks with remaining time budget
            if self._active_tasks:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    logger.warning(
                        "drain: timed out after %.1fs, cancelling %d tasks",
                        effective,
                        len(self._active_tasks),
                    )
                    await self._cancel_active()
                    self._pending.clear()
                    return
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*list(self._active_tasks), return_exceptions=True),
                        timeout=remaining,
                    )
                except TimeoutError:
                    logger.warning(
                        "drain: timed out after %.1fs, cancelling %d tasks",
                        effective,
                        len(self._active_tasks),
                    )
                    await self._cancel_active()
                    self._pending.clear()
                    return

    async def _cancel_active(self) -> None:
        """Cancel all active tasks and wait for them to finish."""
        for task in self._active_tasks:
            task.cancel()
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
        self._active_tasks.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def indexing_mode(self) -> IndexingMode:
        """The current indexing mode."""
        return self._indexing_mode

    @property
    def pending_count(self) -> int:
        """Number of pending (debounced) + active background tasks."""
        return len(self._pending) + len(self._active_tasks)
