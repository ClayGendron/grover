"""PythonAnalyzer — stdlib ast-based analysis."""

from __future__ import annotations

import ast
import posixpath

from grover.graph.analyzers._base import (
    AnalysisResult,
    ChunkFile,
    EdgeData,
    build_chunk_path,
    extract_lines,
)


class PythonAnalyzer:
    """Extracts structure from Python source files using the stdlib ``ast`` module."""

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".py"})

    def analyze_file(self, path: str, content: str) -> AnalysisResult:
        """Parse *content* as Python and extract chunks + edges."""
        if not content.strip():
            return [], []
        try:
            tree = ast.parse(content, filename=path)
        except SyntaxError:
            return [], []

        chunks: list[ChunkFile] = []
        edges: list[EdgeData] = []
        self._visit_body(tree.body, path, content, chunks, edges, scope=[])
        self._collect_imports(tree.body, path, edges)
        return chunks, edges

    def _visit_body(
        self,
        body: list[ast.stmt],
        parent_path: str,
        content: str,
        chunks: list[ChunkFile],
        edges: list[EdgeData],
        scope: list[str],
    ) -> None:
        """Recursively visit a body of statements, extracting functions and classes."""
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                self._handle_function(node, parent_path, content, chunks, edges, scope)
            elif isinstance(node, ast.ClassDef):
                self._handle_class(node, parent_path, content, chunks, edges, scope)

    def _handle_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_path: str,
        content: str,
        chunks: list[ChunkFile],
        edges: list[EdgeData],
        scope: list[str],
    ) -> None:
        scoped_name = ".".join([*scope, node.name])
        chunk_path = build_chunk_path(parent_path, scoped_name)
        line_start = node.lineno
        line_end = node.end_lineno or node.lineno
        chunk_content = extract_lines(content, line_start, line_end)

        chunks.append(
            ChunkFile(
                chunk_path=chunk_path,
                parent_path=parent_path,
                content=chunk_content,
                line_start=line_start,
                line_end=line_end,
                name=scoped_name,
            )
        )
        edges.append(EdgeData(source=parent_path, target=chunk_path, edge_type="contains"))

        # Recurse into nested definitions
        self._visit_body(
            node.body,
            parent_path,
            content,
            chunks,
            edges,
            scope=[*scope, node.name],
        )

    def _handle_class(
        self,
        node: ast.ClassDef,
        parent_path: str,
        content: str,
        chunks: list[ChunkFile],
        edges: list[EdgeData],
        scope: list[str],
    ) -> None:
        scoped_name = ".".join([*scope, node.name])
        chunk_path = build_chunk_path(parent_path, scoped_name)
        line_start = node.lineno
        line_end = node.end_lineno or node.lineno
        chunk_content = extract_lines(content, line_start, line_end)

        chunks.append(
            ChunkFile(
                chunk_path=chunk_path,
                parent_path=parent_path,
                content=chunk_content,
                line_start=line_start,
                line_end=line_end,
                name=scoped_name,
            )
        )
        edges.append(EdgeData(source=parent_path, target=chunk_path, edge_type="contains"))

        # Inheritance edges
        for base in node.bases:
            base_name = self._resolve_base_name(base)
            if base_name:
                edges.append(
                    EdgeData(
                        source=chunk_path,
                        target=base_name,
                        edge_type="inherits",
                        metadata={"base_name": base_name},
                    )
                )

        # Recurse into class body for methods, nested classes
        self._visit_body(
            node.body,
            parent_path,
            content,
            chunks,
            edges,
            scope=[*scope, node.name],
        )

    def _collect_imports(
        self,
        body: list[ast.stmt],
        parent_path: str,
        edges: list[EdgeData],
    ) -> None:
        """Walk the top-level body and emit import edges."""
        parent_dir = posixpath.dirname(parent_path)

        for node in body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = self._module_to_path(alias.name)
                    edges.append(
                        EdgeData(
                            source=parent_path,
                            target=target,
                            edge_type="imports",
                            metadata={"module": alias.name},
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    # Relative import
                    target = self._resolve_relative_import(
                        parent_dir, node.module or "", node.level
                    )
                else:
                    target = self._module_to_path(node.module or "")
                edges.append(
                    EdgeData(
                        source=parent_path,
                        target=target,
                        edge_type="imports",
                        metadata={"module": node.module or "", "level": node.level or 0},
                    )
                )

    @staticmethod
    def _module_to_path(module: str) -> str:
        """Convert a dotted module name to a heuristic file path.

        ``foo.bar`` → ``/foo/bar.py``
        """
        parts = module.split(".")
        return "/" + "/".join(parts) + ".py"

    @staticmethod
    def _resolve_relative_import(parent_dir: str, module: str, level: int) -> str:
        """Resolve a relative import to an absolute path.

        ``from ..utils import foo`` in ``/src/pkg/sub/mod.py`` (level=2)
        → ``/src/pkg/utils.py``
        """
        base = parent_dir
        for _ in range(level - 1):
            base = posixpath.dirname(base)
        if module:
            parts = module.split(".")
            return posixpath.join(base, *parts) + ".py"
        # ``from . import X`` — refers to the package __init__
        return posixpath.join(base, "__init__.py")

    @staticmethod
    def _resolve_base_name(node: ast.expr) -> str | None:
        """Extract a base class name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            # e.g. module.ClassName — just return the full dotted name
            parts: list[str] = []
            current: ast.expr = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return None
