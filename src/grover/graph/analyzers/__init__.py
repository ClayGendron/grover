"""AST analyzers — extract structure from source files."""

from __future__ import annotations

import logging
import posixpath

from grover.graph._rustworkx import RustworkxGraph
from grover.graph.analyzers._base import (
    AnalysisResult,
    Analyzer,
    ChunkFile,
    EdgeData,
    build_chunk_path,
    extract_lines,
)
from grover.graph.analyzers.python import PythonAnalyzer

logger = logging.getLogger(__name__)


class AnalyzerRegistry:
    """Maps file extensions to language-specific analyzers."""

    def __init__(self) -> None:
        self._ext_map: dict[str, Analyzer] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Auto-register built-in analyzers."""
        # Python — always available (stdlib ast)
        self.register(PythonAnalyzer())

        # JS/TS — depends on tree-sitter
        try:
            from grover.graph.analyzers.javascript import (
                JavaScriptAnalyzer,
                TypeScriptAnalyzer,
            )

            self.register(JavaScriptAnalyzer())
            self.register(TypeScriptAnalyzer())
        except Exception:
            logger.debug("JavaScript/TypeScript analyzers not available")

        # Go — depends on tree-sitter
        try:
            from grover.graph.analyzers.go import GoAnalyzer

            self.register(GoAnalyzer())
        except Exception:
            logger.debug("Go analyzer not available")

    def register(self, analyzer: Analyzer) -> None:
        """Register an analyzer for each of its extensions."""
        for ext in analyzer.extensions:
            self._ext_map[ext.lower()] = analyzer

    def get(self, path: str) -> Analyzer | None:
        """Look up an analyzer by file path extension (case-insensitive)."""
        ext = posixpath.splitext(path)[1].lower()
        return self._ext_map.get(ext)

    def supported_extensions(self) -> frozenset[str]:
        """Return all registered extensions."""
        return frozenset(self._ext_map.keys())

    def analyze_file(self, path: str, content: str) -> AnalysisResult | None:
        """Convenience: look up analyzer and run it. Returns ``None`` if unsupported."""
        analyzer = self.get(path)
        if analyzer is None:
            return None
        return analyzer.analyze_file(path, content)


_default_registry = AnalyzerRegistry()


def get_analyzer(path: str) -> Analyzer | None:
    """Get the analyzer for *path* from the default registry."""
    return _default_registry.get(path)


__all__ = [
    "Analyzer",
    "AnalyzerRegistry",
    "ChunkFile",
    "EdgeData",
    "PythonAnalyzer",
    "RustworkxGraph",
    "build_chunk_path",
    "extract_lines",
    "get_analyzer",
]
