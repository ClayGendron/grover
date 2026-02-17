"""GroverMiddleware — deepagents AgentMiddleware exposing Grover-specific tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.agents.middleware.types import AgentMiddleware

if TYPE_CHECKING:
    from grover._grover import Grover


class GroverMiddleware(AgentMiddleware):
    """deepagents middleware exposing Grover version, search, graph, and trash tools.

    Placeholder — full implementation in Phase 3.
    """

    def __init__(
        self,
        grover: Grover,
        *,
        enable_search: bool = True,
        enable_graph: bool = True,
    ) -> None:
        self.grover = grover
        self.tools: list = []
