"""Grover integration with deepagents (LangGraph agent framework).

Provides ``GroverBackend`` (``BackendProtocol`` implementation) and
``GroverMiddleware`` (version/search/graph/trash tools) so any LangGraph
deep agent can use Grover as its storage backend.

Usage::

    from grover import Grover
    from grover.fs.local_fs import LocalFileSystem
    from grover.integrations.deepagents import GroverBackend, GroverMiddleware

    g = Grover()
    g.mount("/project", LocalFileSystem(workspace_dir="/tmp/test"))

    backend = GroverBackend(g)
    middleware = GroverMiddleware(g)
"""

try:
    from deepagents.backends.protocol import BackendProtocol as _BackendProtocol
except ImportError as _exc:
    raise ImportError(
        "deepagents is required for the Grover deepagents integration. "
        "Install it with: pip install 'grover[deepagents]'"
    ) from _exc

from grover.integrations.deepagents._backend import GroverBackend
from grover.integrations.deepagents._middleware import GroverMiddleware

__all__ = ["GroverBackend", "GroverMiddleware"]
