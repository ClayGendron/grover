"""GoAnalyzer — tree-sitter-based analysis."""

from __future__ import annotations

import logging

from grover.graph.analyzers._base import (
    AnalysisResult,
    ChunkFile,
    EdgeData,
    build_chunk_path,
    extract_lines,
)

logger = logging.getLogger(__name__)

try:
    import tree_sitter
    from tree_sitter_go import language as _go_language

    _HAS_TREESITTER = True
except ImportError:  # pragma: no cover
    _HAS_TREESITTER = False


class GoAnalyzer:
    """Extracts structure from Go source files using tree-sitter."""

    _warned: bool = False

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".go"})

    def analyze_file(self, path: str, content: str) -> AnalysisResult:
        if not _HAS_TREESITTER:
            if not GoAnalyzer._warned:
                logger.warning("tree-sitter not available; GoAnalyzer returning empty results")
                GoAnalyzer._warned = True
            return [], []
        if not content.strip():
            return [], []

        lang = tree_sitter.Language(_go_language())
        parser = tree_sitter.Parser(lang)
        tree = parser.parse(content.encode())
        root = tree.root_node

        chunks: list[ChunkFile] = []
        edges: list[EdgeData] = []

        for child in root.children:
            if child.type == "function_declaration":
                _handle_function(child, path, content, chunks, edges)
            elif child.type == "method_declaration":
                _handle_method(child, path, content, chunks, edges)
            elif child.type == "type_declaration":
                _handle_type_decl(child, path, content, chunks, edges)
            elif child.type == "import_declaration":
                _collect_imports(child, path, edges)

        return chunks, edges


def _handle_function(
    node: tree_sitter.Node,
    parent_path: str,
    content: str,
    chunks: list[ChunkFile],
    edges: list[EdgeData],
) -> None:
    name_node = node.child_by_field_name("name")
    if not name_node or name_node.text is None:
        return
    name = name_node.text.decode()

    # Skip init functions (special, can duplicate)
    if name == "init":
        return

    chunk_path = build_chunk_path(parent_path, name)
    line_start = node.start_point.row + 1
    line_end = node.end_point.row + 1
    chunk_content = extract_lines(content, line_start, line_end)

    chunks.append(
        ChunkFile(
            chunk_path=chunk_path,
            parent_path=parent_path,
            content=chunk_content,
            line_start=line_start,
            line_end=line_end,
            name=name,
        )
    )
    edges.append(EdgeData(source=parent_path, target=chunk_path, edge_type="contains"))


def _handle_method(
    node: tree_sitter.Node,
    parent_path: str,
    content: str,
    chunks: list[ChunkFile],
    edges: list[EdgeData],
) -> None:
    """Handle ``func (r *Receiver) Method()`` — scoped as ``Receiver.Method``."""
    name_node = node.child_by_field_name("name")
    if not name_node or name_node.text is None:
        return
    method_name = name_node.text.decode()

    receiver_type = _extract_receiver_type(node)
    scoped_name = f"{receiver_type}.{method_name}" if receiver_type else method_name

    chunk_path = build_chunk_path(parent_path, scoped_name)
    line_start = node.start_point.row + 1
    line_end = node.end_point.row + 1
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
    if receiver_type:
        edges.append(
            EdgeData(
                source=chunk_path,
                target=chunk_path,
                edge_type="method_of",
                metadata={"receiver": receiver_type},
            )
        )


def _extract_receiver_type(node: tree_sitter.Node) -> str | None:
    """Extract the receiver type name from a method_declaration.

    Handles both ``(s Server)`` and ``(s *Server)`` forms.
    """
    receiver = node.child_by_field_name("receiver")
    if not receiver:
        return None
    for param in receiver.children:
        if param.type != "parameter_declaration":
            continue
        type_node = param.child_by_field_name("type")
        if not type_node:
            continue
        if type_node.type == "pointer_type":
            # *Server → find the type_identifier inside
            for inner in type_node.children:
                if inner.type == "type_identifier" and inner.text is not None:
                    return inner.text.decode()
        elif type_node.type == "type_identifier" and type_node.text is not None:
            return type_node.text.decode()
    return None


def _handle_type_decl(
    node: tree_sitter.Node,
    parent_path: str,
    content: str,
    chunks: list[ChunkFile],
    edges: list[EdgeData],
) -> None:
    """Handle ``type X struct { ... }`` and ``type X interface { ... }``."""
    for child in node.children:
        if child.type != "type_spec":
            continue
        name_node = child.child_by_field_name("name")
        if not name_node or name_node.text is None:
            continue
        name = name_node.text.decode()

        chunk_path = build_chunk_path(parent_path, name)
        line_start = node.start_point.row + 1
        line_end = node.end_point.row + 1
        chunk_content = extract_lines(content, line_start, line_end)

        # Determine kind from the type field
        type_field = child.child_by_field_name("type")
        kind = type_field.type if type_field else "unknown"

        chunks.append(
            ChunkFile(
                chunk_path=chunk_path,
                parent_path=parent_path,
                content=chunk_content,
                line_start=line_start,
                line_end=line_end,
                name=name,
            )
        )
        edges.append(
            EdgeData(
                source=parent_path,
                target=chunk_path,
                edge_type="contains",
                metadata={"kind": kind},
            )
        )


def _collect_imports(
    node: tree_sitter.Node,
    parent_path: str,
    edges: list[EdgeData],
) -> None:
    """Collect import paths from an import_declaration."""
    for child in node.children:
        if child.type == "import_spec":
            _add_import_edge(child, parent_path, edges)
        elif child.type == "import_spec_list":
            for spec in child.children:
                if spec.type == "import_spec":
                    _add_import_edge(spec, parent_path, edges)


def _add_import_edge(
    spec: tree_sitter.Node,
    parent_path: str,
    edges: list[EdgeData],
) -> None:
    path_node = spec.child_by_field_name("path")
    if not path_node or path_node.text is None:
        return
    # Strip quotes from the interpreted_string_literal
    raw = path_node.text.decode().strip('"')
    edges.append(
        EdgeData(
            source=parent_path,
            target=raw,
            edge_type="imports",
            metadata={"module": raw},
        )
    )
