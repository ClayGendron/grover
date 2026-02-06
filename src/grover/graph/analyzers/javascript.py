"""JavaScriptAnalyzer — tree-sitter-based analysis."""

from __future__ import annotations

import logging
import posixpath

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
    from tree_sitter_javascript import language as _js_language
    from tree_sitter_typescript import language_tsx as _tsx_language
    from tree_sitter_typescript import language_typescript as _ts_language

    _HAS_TREESITTER = True
except ImportError:  # pragma: no cover
    _HAS_TREESITTER = False


class JavaScriptAnalyzer:
    """Extracts structure from JavaScript source files using tree-sitter."""

    _warned: bool = False

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".js", ".jsx", ".mjs", ".cjs"})

    def analyze_file(self, path: str, content: str) -> AnalysisResult:
        if not _HAS_TREESITTER:
            if not JavaScriptAnalyzer._warned:
                logger.warning(
                    "tree-sitter not available; JavaScriptAnalyzer returning empty results"
                )
                JavaScriptAnalyzer._warned = True
            return [], []
        if not content.strip():
            return [], []

        lang = tree_sitter.Language(_js_language())
        return _analyze_js_tree(path, content, lang)


class TypeScriptAnalyzer:
    """Extracts structure from TypeScript source files using tree-sitter.

    Reuses the same JS analysis logic — TypeScript and JavaScript share
    the same tree-sitter AST structure for declarations.
    """

    _warned: bool = False

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".ts", ".tsx"})

    def analyze_file(self, path: str, content: str) -> AnalysisResult:
        if not _HAS_TREESITTER:
            if not TypeScriptAnalyzer._warned:
                logger.warning(
                    "tree-sitter not available; TypeScriptAnalyzer returning empty results"
                )
                TypeScriptAnalyzer._warned = True
            return [], []
        if not content.strip():
            return [], []

        if path.endswith(".tsx"):
            lang = tree_sitter.Language(_tsx_language())
        else:
            lang = tree_sitter.Language(_ts_language())
        return _analyze_js_tree(path, content, lang)


def _analyze_js_tree(
    path: str,
    content: str,
    lang: tree_sitter.Language,
) -> AnalysisResult:
    """Shared analysis logic for JS/TS ASTs."""
    parser = tree_sitter.Parser(lang)
    tree = parser.parse(content.encode())
    root = tree.root_node

    chunks: list[ChunkFile] = []
    edges: list[EdgeData] = []

    _visit_children(root, path, content, chunks, edges, scope=[])
    _collect_imports(root, path, edges)
    return chunks, edges


def _visit_children(
    node: tree_sitter.Node,
    parent_path: str,
    content: str,
    chunks: list[ChunkFile],
    edges: list[EdgeData],
    scope: list[str],
) -> None:
    """Walk direct children looking for declarations."""
    for child in node.children:
        if child.type == "function_declaration":
            _handle_function(child, parent_path, content, chunks, edges, scope)
        elif child.type == "class_declaration":
            _handle_class(child, parent_path, content, chunks, edges, scope)
        elif child.type == "lexical_declaration":
            _handle_lexical(child, parent_path, content, chunks, edges, scope)
        elif child.type == "export_statement":
            # Unwrap export: recurse into its children for the actual declaration
            _visit_children(child, parent_path, content, chunks, edges, scope)


def _node_name(node: tree_sitter.Node) -> str | None:
    """Extract the name from a declaration node via the 'name' field."""
    name_node = node.child_by_field_name("name")
    if name_node and name_node.text is not None:
        return name_node.text.decode()
    return None


def _make_chunk(
    node: tree_sitter.Node,
    name: str,
    parent_path: str,
    content: str,
    scope: list[str],
) -> ChunkFile:
    scoped_name = ".".join([*scope, name])
    chunk_path = build_chunk_path(parent_path, scoped_name)
    line_start = node.start_point.row + 1
    line_end = node.end_point.row + 1
    chunk_content = extract_lines(content, line_start, line_end)
    return ChunkFile(
        chunk_path=chunk_path,
        parent_path=parent_path,
        content=chunk_content,
        line_start=line_start,
        line_end=line_end,
        name=scoped_name,
    )


def _handle_function(
    node: tree_sitter.Node,
    parent_path: str,
    content: str,
    chunks: list[ChunkFile],
    edges: list[EdgeData],
    scope: list[str],
) -> None:
    name = _node_name(node)
    if not name:
        return
    chunk = _make_chunk(node, name, parent_path, content, scope)
    chunks.append(chunk)
    edges.append(EdgeData(source=parent_path, target=chunk.chunk_path, edge_type="contains"))


def _handle_class(
    node: tree_sitter.Node,
    parent_path: str,
    content: str,
    chunks: list[ChunkFile],
    edges: list[EdgeData],
    scope: list[str],
) -> None:
    name = _node_name(node)
    if not name:
        return
    chunk = _make_chunk(node, name, parent_path, content, scope)
    chunks.append(chunk)
    edges.append(EdgeData(source=parent_path, target=chunk.chunk_path, edge_type="contains"))

    # Inheritance from class_heritage
    for child in node.children:
        if child.type == "class_heritage":
            for hc in child.children:
                if hc.is_named and hc.type == "identifier" and hc.text is not None:
                    base_name = hc.text.decode()
                    edges.append(
                        EdgeData(
                            source=chunk.chunk_path,
                            target=base_name,
                            edge_type="inherits",
                            metadata={"base_name": base_name},
                        )
                    )

    # Methods inside class_body
    body = node.child_by_field_name("body")
    if body:
        for child in body.children:
            if child.type == "method_definition":
                _handle_method(child, parent_path, content, chunks, edges, [*scope, name])


def _handle_method(
    node: tree_sitter.Node,
    parent_path: str,
    content: str,
    chunks: list[ChunkFile],
    edges: list[EdgeData],
    scope: list[str],
) -> None:
    name = _node_name(node)
    if not name:
        return
    chunk = _make_chunk(node, name, parent_path, content, scope)
    chunks.append(chunk)
    edges.append(EdgeData(source=parent_path, target=chunk.chunk_path, edge_type="contains"))


def _handle_lexical(
    node: tree_sitter.Node,
    parent_path: str,
    content: str,
    chunks: list[ChunkFile],
    edges: list[EdgeData],
    scope: list[str],
) -> None:
    """Handle ``const/let`` declarations — only extract arrow functions / function expressions."""
    for child in node.children:
        if child.type != "variable_declarator":
            continue
        name = _node_name(child)
        value = child.child_by_field_name("value")
        if not name or not value:
            continue
        if value.type not in ("arrow_function", "function_expression"):
            continue
        # Use the full lexical_declaration node for line range
        chunk = _make_chunk(node, name, parent_path, content, scope)
        chunks.append(chunk)
        edges.append(EdgeData(source=parent_path, target=chunk.chunk_path, edge_type="contains"))


def _collect_imports(
    root: tree_sitter.Node,
    parent_path: str,
    edges: list[EdgeData],
) -> None:
    """Collect import statements from root children."""
    parent_dir = posixpath.dirname(parent_path)

    for child in root.children:
        if child.type == "import_statement":
            source_node = child.child_by_field_name("source")
            if source_node and source_node.text is not None:
                raw = source_node.text.decode().strip("'\"")
                target = _resolve_js_import(parent_dir, raw)
                edges.append(
                    EdgeData(
                        source=parent_path,
                        target=target,
                        edge_type="imports",
                        metadata={"module": raw},
                    )
                )
        elif child.type == "export_statement":
            # Re-export: export { x } from './foo'
            source_node = child.child_by_field_name("source")
            if source_node and source_node.text is not None:
                raw = source_node.text.decode().strip("'\"")
                target = _resolve_js_import(parent_dir, raw)
                edges.append(
                    EdgeData(
                        source=parent_path,
                        target=target,
                        edge_type="imports",
                        metadata={"module": raw},
                    )
                )


def _resolve_js_import(parent_dir: str, raw: str) -> str:
    """Resolve a JS/TS import specifier to a heuristic file path."""
    if raw.startswith("."):
        # Relative import
        resolved = posixpath.normpath(posixpath.join(parent_dir, raw))
        if not posixpath.splitext(resolved)[1]:
            resolved += ".js"
        return resolved
    # Bare specifier → node_modules
    return f"/node_modules/{raw}.js"
