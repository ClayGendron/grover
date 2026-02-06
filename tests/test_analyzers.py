"""Tests for AST analyzers — Python, JavaScript, TypeScript, Go."""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from grover.graph.analyzers import (
    Analyzer,
    AnalyzerRegistry,
    ChunkFile,
    EdgeData,
    PythonAnalyzer,
    build_chunk_path,
    extract_lines,
    get_analyzer,
)

# ---------------------------------------------------------------------------
# Fixtures / constants
# ---------------------------------------------------------------------------

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "repos"


# ===================================================================
# TestChunkFile
# ===================================================================


class TestChunkFile:
    def test_construction(self):
        c = ChunkFile(
            chunk_path="/.grover/chunks/src/foo_py/bar.txt",
            parent_path="/src/foo.py",
            content="def bar(): pass",
            line_start=1,
            line_end=1,
            name="bar",
        )
        assert c.chunk_path == "/.grover/chunks/src/foo_py/bar.txt"
        assert c.parent_path == "/src/foo.py"
        assert c.name == "bar"

    def test_frozen(self):
        c = ChunkFile(
            chunk_path="x", parent_path="y", content="z",
            line_start=1, line_end=1, name="n",
        )
        with pytest.raises(AttributeError):
            c.name = "other"  # type: ignore[misc]


# ===================================================================
# TestEdgeData
# ===================================================================


class TestEdgeData:
    def test_construction(self):
        e = EdgeData(source="/a.py", target="/b.py", edge_type="imports")
        assert e.source == "/a.py"
        assert e.target == "/b.py"
        assert e.edge_type == "imports"

    def test_default_metadata(self):
        e = EdgeData(source="/a", target="/b", edge_type="x")
        assert e.metadata == {}

    def test_frozen(self):
        e = EdgeData(source="/a", target="/b", edge_type="x")
        with pytest.raises(AttributeError):
            e.source = "/c"  # type: ignore[misc]


# ===================================================================
# TestBuildChunkPath
# ===================================================================


class TestBuildChunkPath:
    def test_simple_function(self):
        assert build_chunk_path("/src/auth.py", "login") == (
            "/.grover/chunks/src/auth_py/login.txt"
        )

    def test_scoped_name(self):
        assert build_chunk_path("/src/auth.py", "Client.connect") == (
            "/.grover/chunks/src/auth_py/Client.connect.txt"
        )

    def test_root_file(self):
        assert build_chunk_path("/main.py", "run") == (
            "/.grover/chunks/main_py/run.txt"
        )

    def test_deeply_nested_dir(self):
        assert build_chunk_path("/a/b/c/d.py", "foo") == (
            "/.grover/chunks/a/b/c/d_py/foo.txt"
        )

    def test_dotted_filename(self):
        assert build_chunk_path("/src/auth.test.py", "test_login") == (
            "/.grover/chunks/src/auth_test_py/test_login.txt"
        )

    def test_dunder_method(self):
        assert build_chunk_path("/src/cls.py", "MyClass.__init__") == (
            "/.grover/chunks/src/cls_py/MyClass.__init__.txt"
        )


# ===================================================================
# TestExtractLines
# ===================================================================


class TestExtractLines:
    def test_single_line(self):
        content = "line1\nline2\nline3\n"
        assert extract_lines(content, 2, 2) == "line2\n"

    def test_range(self):
        content = "a\nb\nc\nd\n"
        assert extract_lines(content, 2, 3) == "b\nc\n"

    def test_clamps_bounds(self):
        content = "only\n"
        result = extract_lines(content, 1, 100)
        assert result == "only\n"

    def test_preserves_indentation(self):
        content = "def foo():\n    return 1\n"
        assert extract_lines(content, 2, 2) == "    return 1\n"


# ===================================================================
# TestPythonFunctions
# ===================================================================


class TestPythonFunctions:
    SAMPLE = textwrap.dedent("""\
        def login(user):
            return True

        def logout():
            pass
    """)

    def test_extracts_top_level_functions(self):
        analyzer = PythonAnalyzer()
        chunks = analyzer.analyze_file("/src/auth.py", self.SAMPLE)[0]
        names = [c.name for c in chunks]
        assert "login" in names
        assert "logout" in names

    def test_chunk_content_matches(self):
        analyzer = PythonAnalyzer()
        chunks = analyzer.analyze_file("/src/auth.py", self.SAMPLE)[0]
        login_chunk = next(c for c in chunks if c.name == "login")
        assert "def login(user):" in login_chunk.content
        assert "return True" in login_chunk.content

    def test_line_numbers(self):
        analyzer = PythonAnalyzer()
        chunks = analyzer.analyze_file("/src/auth.py", self.SAMPLE)[0]
        login_chunk = next(c for c in chunks if c.name == "login")
        assert login_chunk.line_start == 1
        assert login_chunk.line_end == 2

    def test_contains_edges(self):
        analyzer = PythonAnalyzer()
        edges = analyzer.analyze_file("/src/auth.py", self.SAMPLE)[1]
        contains = [e for e in edges if e.edge_type == "contains"]
        assert len(contains) == 2
        assert all(e.source == "/src/auth.py" for e in contains)

    def test_stable_paths(self):
        analyzer = PythonAnalyzer()
        chunks = analyzer.analyze_file("/src/auth.py", self.SAMPLE)[0]
        login_chunk = next(c for c in chunks if c.name == "login")
        assert login_chunk.chunk_path == "/.grover/chunks/src/auth_py/login.txt"


# ===================================================================
# TestPythonClasses
# ===================================================================


class TestPythonClasses:
    SAMPLE = textwrap.dedent("""\
        class Client:
            def connect(self):
                pass

            def disconnect(self):
                pass
    """)

    def test_class_and_methods(self):
        analyzer = PythonAnalyzer()
        chunks = analyzer.analyze_file("/src/client.py", self.SAMPLE)[0]
        names = [c.name for c in chunks]
        assert "Client" in names
        assert "Client.connect" in names
        assert "Client.disconnect" in names

    def test_scoped_names(self):
        analyzer = PythonAnalyzer()
        chunks = analyzer.analyze_file("/src/client.py", self.SAMPLE)[0]
        connect = next(c for c in chunks if c.name == "Client.connect")
        assert connect.chunk_path == "/.grover/chunks/src/client_py/Client.connect.txt"

    def test_class_contains_edge(self):
        analyzer = PythonAnalyzer()
        edges = analyzer.analyze_file("/src/client.py", self.SAMPLE)[1]
        contains = [e for e in edges if e.edge_type == "contains"]
        # 1 for class + 2 for methods = 3
        assert len(contains) == 3

    def test_method_contains_edges(self):
        analyzer = PythonAnalyzer()
        edges = analyzer.analyze_file("/src/client.py", self.SAMPLE)[1]
        contains = [e for e in edges if e.edge_type == "contains"]
        targets = [e.target for e in contains]
        assert any("Client.connect" in t for t in targets)
        assert any("Client.disconnect" in t for t in targets)


# ===================================================================
# TestPythonNested
# ===================================================================


class TestPythonNested:
    SAMPLE = textwrap.dedent("""\
        class Outer:
            class Inner:
                def method(self):
                    pass

        def top():
            def helper():
                pass
    """)

    def test_nested_class(self):
        analyzer = PythonAnalyzer()
        chunks = analyzer.analyze_file("/src/nested.py", self.SAMPLE)[0]
        names = [c.name for c in chunks]
        assert "Outer.Inner" in names
        assert "Outer.Inner.method" in names

    def test_nested_function(self):
        analyzer = PythonAnalyzer()
        chunks = analyzer.analyze_file("/src/nested.py", self.SAMPLE)[0]
        names = [c.name for c in chunks]
        assert "top.helper" in names

    def test_correct_parent_path(self):
        analyzer = PythonAnalyzer()
        chunks = analyzer.analyze_file("/src/nested.py", self.SAMPLE)[0]
        for chunk in chunks:
            assert chunk.parent_path == "/src/nested.py"

    def test_deep_scoping(self):
        analyzer = PythonAnalyzer()
        chunks = analyzer.analyze_file("/src/nested.py", self.SAMPLE)[0]
        inner_method = next(c for c in chunks if c.name == "Outer.Inner.method")
        assert "Outer.Inner.method" in inner_method.chunk_path


# ===================================================================
# TestPythonInheritance
# ===================================================================


class TestPythonInheritance:
    SAMPLE = textwrap.dedent("""\
        class Base:
            pass

        class Child(Base):
            pass

        class Multi(Base, Mixin):
            pass
    """)

    def test_inherits_edges(self):
        analyzer = PythonAnalyzer()
        edges = analyzer.analyze_file("/src/inh.py", self.SAMPLE)[1]
        inherits = [e for e in edges if e.edge_type == "inherits"]
        assert len(inherits) == 3  # Child->Base, Multi->Base, Multi->Mixin

    def test_multiple_inheritance(self):
        analyzer = PythonAnalyzer()
        edges = analyzer.analyze_file("/src/inh.py", self.SAMPLE)[1]
        inherits = [e for e in edges if e.edge_type == "inherits"]
        targets = [e.target for e in inherits]
        assert "Base" in targets
        assert "Mixin" in targets

    def test_source_is_chunk_path(self):
        analyzer = PythonAnalyzer()
        edges = analyzer.analyze_file("/src/inh.py", self.SAMPLE)[1]
        inherits = [e for e in edges if e.edge_type == "inherits"]
        for e in inherits:
            assert e.source.startswith("/.grover/chunks/")

    def test_metadata(self):
        analyzer = PythonAnalyzer()
        edges = analyzer.analyze_file("/src/inh.py", self.SAMPLE)[1]
        inherits = [e for e in edges if e.edge_type == "inherits"]
        for e in inherits:
            assert "base_name" in e.metadata


# ===================================================================
# TestPythonImports
# ===================================================================


class TestPythonImports:
    def test_import_statement(self):
        code = "import os\n"
        analyzer = PythonAnalyzer()
        edges = analyzer.analyze_file("/src/mod.py", code)[1]
        imports = [e for e in edges if e.edge_type == "imports"]
        assert len(imports) == 1
        assert imports[0].target == "/os.py"

    def test_from_import(self):
        code = "from os.path import join\n"
        analyzer = PythonAnalyzer()
        edges = analyzer.analyze_file("/src/mod.py", code)[1]
        imports = [e for e in edges if e.edge_type == "imports"]
        assert imports[0].target == "/os/path.py"

    def test_dotted_import(self):
        code = "import foo.bar.baz\n"
        analyzer = PythonAnalyzer()
        edges = analyzer.analyze_file("/src/mod.py", code)[1]
        imports = [e for e in edges if e.edge_type == "imports"]
        assert imports[0].target == "/foo/bar/baz.py"

    def test_relative_import(self):
        code = "from .utils import helper\n"
        analyzer = PythonAnalyzer()
        edges = analyzer.analyze_file("/src/pkg/mod.py", code)[1]
        imports = [e for e in edges if e.edge_type == "imports"]
        assert imports[0].target == "/src/pkg/utils.py"

    def test_parent_relative_import(self):
        code = "from ..core import base\n"
        analyzer = PythonAnalyzer()
        edges = analyzer.analyze_file("/src/pkg/sub/mod.py", code)[1]
        imports = [e for e in edges if e.edge_type == "imports"]
        assert imports[0].target == "/src/pkg/core.py"

    def test_import_metadata(self):
        code = "from os.path import join\n"
        analyzer = PythonAnalyzer()
        edges = analyzer.analyze_file("/src/mod.py", code)[1]
        imports = [e for e in edges if e.edge_type == "imports"]
        assert imports[0].metadata["module"] == "os.path"


# ===================================================================
# TestPythonEdgeCases
# ===================================================================


class TestPythonEdgeCases:
    def test_syntax_error_returns_empty(self):
        code = "def broken(\n"
        analyzer = PythonAnalyzer()
        chunks, edges = analyzer.analyze_file("/src/bad.py", code)
        assert chunks == []
        assert edges == []

    def test_empty_content_returns_empty(self):
        analyzer = PythonAnalyzer()
        chunks, edges = analyzer.analyze_file("/src/empty.py", "")
        assert chunks == []
        assert edges == []

    def test_satisfies_protocol(self):
        assert isinstance(PythonAnalyzer(), Analyzer)


# ===================================================================
# TestJSStructures
# ===================================================================


class TestJSStructures:
    def test_function_declaration(self):
        from grover.graph.analyzers.javascript import JavaScriptAnalyzer

        code = "function greet(name) { return 'hello ' + name; }\n"
        analyzer = JavaScriptAnalyzer()
        chunks = analyzer.analyze_file("/src/app.js", code)[0]
        names = [c.name for c in chunks]
        assert "greet" in names

    def test_arrow_function(self):
        from grover.graph.analyzers.javascript import JavaScriptAnalyzer

        code = "const add = (a, b) => a + b;\n"
        analyzer = JavaScriptAnalyzer()
        chunks = analyzer.analyze_file("/src/util.js", code)[0]
        names = [c.name for c in chunks]
        assert "add" in names

    def test_class_and_methods(self):
        from grover.graph.analyzers.javascript import JavaScriptAnalyzer

        code = textwrap.dedent("""\
            class Greeter {
                constructor(name) {
                    this.name = name;
                }
                greet() {
                    return 'hello ' + this.name;
                }
            }
        """)
        analyzer = JavaScriptAnalyzer()
        chunks = analyzer.analyze_file("/src/greeter.js", code)[0]
        names = [c.name for c in chunks]
        assert "Greeter" in names
        assert "Greeter.constructor" in names
        assert "Greeter.greet" in names

    def test_inheritance(self):
        from grover.graph.analyzers.javascript import JavaScriptAnalyzer

        code = "class Dog extends Animal { bark() {} }\n"
        analyzer = JavaScriptAnalyzer()
        edges = analyzer.analyze_file("/src/dog.js", code)[1]
        inherits = [e for e in edges if e.edge_type == "inherits"]
        assert len(inherits) == 1
        assert inherits[0].target == "Animal"

    def test_imports(self):
        from grover.graph.analyzers.javascript import JavaScriptAnalyzer

        code = textwrap.dedent("""\
            import { foo } from './utils';
            import lodash from 'lodash';
        """)
        analyzer = JavaScriptAnalyzer()
        edges = analyzer.analyze_file("/src/app.js", code)[1]
        imports = [e for e in edges if e.edge_type == "imports"]
        targets = [e.target for e in imports]
        assert "/src/utils.js" in targets
        assert "/node_modules/lodash.js" in targets

    def test_contains_edges(self):
        from grover.graph.analyzers.javascript import JavaScriptAnalyzer

        code = "function a() {}\nfunction b() {}\n"
        analyzer = JavaScriptAnalyzer()
        edges = analyzer.analyze_file("/src/app.js", code)[1]
        contains = [e for e in edges if e.edge_type == "contains"]
        assert len(contains) == 2
        assert all(e.source == "/src/app.js" for e in contains)

    def test_exported_function(self):
        from grover.graph.analyzers.javascript import JavaScriptAnalyzer

        code = "export function handler(req, res) {}\n"
        analyzer = JavaScriptAnalyzer()
        chunks = analyzer.analyze_file("/src/api.js", code)[0]
        names = [c.name for c in chunks]
        assert "handler" in names


# ===================================================================
# TestJSGraceful
# ===================================================================


class TestJSGraceful:
    def test_returns_empty_without_treesitter(self):
        from grover.graph.analyzers import javascript

        orig = javascript._HAS_TREESITTER
        javascript._HAS_TREESITTER = False
        javascript.JavaScriptAnalyzer._warned = False
        try:
            analyzer = javascript.JavaScriptAnalyzer()
            chunks, edges = analyzer.analyze_file("/src/app.js", "function foo() {}")
            assert chunks == []
            assert edges == []
        finally:
            javascript._HAS_TREESITTER = orig
            javascript.JavaScriptAnalyzer._warned = False

    def test_logs_warning_once(self, caplog):
        from grover.graph.analyzers import javascript

        orig = javascript._HAS_TREESITTER
        javascript._HAS_TREESITTER = False
        javascript.JavaScriptAnalyzer._warned = False
        try:
            analyzer = javascript.JavaScriptAnalyzer()
            with caplog.at_level("WARNING", logger="grover.graph.analyzers.javascript"):
                analyzer.analyze_file("/a.js", "x")
                analyzer.analyze_file("/b.js", "y")
            warnings = [r for r in caplog.records if r.levelname == "WARNING"]
            assert len(warnings) == 1
        finally:
            javascript._HAS_TREESITTER = orig
            javascript.JavaScriptAnalyzer._warned = False


# ===================================================================
# TestTSAnalyzer
# ===================================================================


class TestTSAnalyzer:
    def test_extensions(self):
        from grover.graph.analyzers.javascript import TypeScriptAnalyzer

        ts = TypeScriptAnalyzer()
        assert ".ts" in ts.extensions
        assert ".tsx" in ts.extensions

    def test_satisfies_protocol(self):
        from grover.graph.analyzers.javascript import TypeScriptAnalyzer

        assert isinstance(TypeScriptAnalyzer(), Analyzer)

    def test_analyzes_typescript(self):
        from grover.graph.analyzers.javascript import TypeScriptAnalyzer

        code = textwrap.dedent("""\
            function greet(name: string): string {
                return 'hello ' + name;
            }
        """)
        ts = TypeScriptAnalyzer()
        chunks = ts.analyze_file("/src/app.ts", code)[0]
        names = [c.name for c in chunks]
        assert "greet" in names


# ===================================================================
# TestGoStructures
# ===================================================================


class TestGoStructures:
    def test_function(self):
        from grover.graph.analyzers.go import GoAnalyzer

        code = textwrap.dedent("""\
            package main

            func main() {}
            func helper() string { return "" }
        """)
        analyzer = GoAnalyzer()
        chunks = analyzer.analyze_file("/cmd/main.go", code)[0]
        names = [c.name for c in chunks]
        assert "main" in names
        assert "helper" in names

    def test_type_declaration(self):
        from grover.graph.analyzers.go import GoAnalyzer

        code = textwrap.dedent("""\
            package main

            type Server struct {
                addr string
            }
        """)
        analyzer = GoAnalyzer()
        chunks = analyzer.analyze_file("/pkg/server.go", code)[0]
        names = [c.name for c in chunks]
        assert "Server" in names

    def test_method_with_receiver(self):
        from grover.graph.analyzers.go import GoAnalyzer

        code = textwrap.dedent("""\
            package main

            type Server struct{}
            func (s *Server) Start() {}
        """)
        analyzer = GoAnalyzer()
        chunks = analyzer.analyze_file("/pkg/server.go", code)[0]
        names = [c.name for c in chunks]
        assert "Server.Start" in names

    def test_imports_single(self):
        from grover.graph.analyzers.go import GoAnalyzer

        code = textwrap.dedent("""\
            package main
            import "fmt"
        """)
        analyzer = GoAnalyzer()
        edges = analyzer.analyze_file("/cmd/main.go", code)[1]
        imports = [e for e in edges if e.edge_type == "imports"]
        assert len(imports) == 1
        assert imports[0].target == "fmt"

    def test_imports_grouped(self):
        from grover.graph.analyzers.go import GoAnalyzer

        code = textwrap.dedent("""\
            package main
            import (
                "fmt"
                "net/http"
            )
        """)
        analyzer = GoAnalyzer()
        edges = analyzer.analyze_file("/cmd/main.go", code)[1]
        imports = [e for e in edges if e.edge_type == "imports"]
        targets = [e.target for e in imports]
        assert "fmt" in targets
        assert "net/http" in targets

    def test_contains_edges(self):
        from grover.graph.analyzers.go import GoAnalyzer

        code = textwrap.dedent("""\
            package main
            func a() {}
            func b() {}
        """)
        analyzer = GoAnalyzer()
        edges = analyzer.analyze_file("/main.go", code)[1]
        contains = [e for e in edges if e.edge_type == "contains"]
        assert len(contains) == 2

    def test_method_metadata(self):
        from grover.graph.analyzers.go import GoAnalyzer

        code = textwrap.dedent("""\
            package main
            type S struct{}
            func (s *S) Run() {}
        """)
        analyzer = GoAnalyzer()
        edges = analyzer.analyze_file("/pkg/s.go", code)[1]
        method_of = [e for e in edges if e.edge_type == "method_of"]
        assert len(method_of) == 1
        assert method_of[0].metadata["receiver"] == "S"

    def test_skips_init(self):
        from grover.graph.analyzers.go import GoAnalyzer

        code = textwrap.dedent("""\
            package main
            func init() {}
            func main() {}
        """)
        analyzer = GoAnalyzer()
        chunks = analyzer.analyze_file("/main.go", code)[0]
        names = [c.name for c in chunks]
        assert "init" not in names
        assert "main" in names


# ===================================================================
# TestGoGraceful
# ===================================================================


class TestGoGraceful:
    def test_returns_empty_without_treesitter(self):
        from grover.graph.analyzers import go

        orig = go._HAS_TREESITTER
        go._HAS_TREESITTER = False
        go.GoAnalyzer._warned = False
        try:
            analyzer = go.GoAnalyzer()
            chunks, edges = analyzer.analyze_file("/main.go", "package main\nfunc main() {}")
            assert chunks == []
            assert edges == []
        finally:
            go._HAS_TREESITTER = orig
            go.GoAnalyzer._warned = False

    def test_logs_warning_once(self, caplog):
        from grover.graph.analyzers import go

        orig = go._HAS_TREESITTER
        go._HAS_TREESITTER = False
        go.GoAnalyzer._warned = False
        try:
            analyzer = go.GoAnalyzer()
            with caplog.at_level("WARNING", logger="grover.graph.analyzers.go"):
                analyzer.analyze_file("/a.go", "package main")
                analyzer.analyze_file("/b.go", "package main")
            warnings = [r for r in caplog.records if r.levelname == "WARNING"]
            assert len(warnings) == 1
        finally:
            go._HAS_TREESITTER = orig
            go.GoAnalyzer._warned = False


# ===================================================================
# TestRegistry
# ===================================================================


class TestRegistry:
    def test_python_registered(self):
        reg = AnalyzerRegistry()
        assert reg.get("/foo.py") is not None

    def test_get_by_extension(self):
        reg = AnalyzerRegistry()
        analyzer = reg.get("/src/app.js")
        assert analyzer is not None
        assert ".js" in analyzer.extensions

    def test_unsupported_returns_none(self):
        reg = AnalyzerRegistry()
        assert reg.get("/data.csv") is None

    def test_supported_extensions(self):
        reg = AnalyzerRegistry()
        exts = reg.supported_extensions()
        assert ".py" in exts
        assert ".js" in exts
        assert ".go" in exts

    def test_custom_analyzer(self):
        class RustAnalyzer:
            @property
            def extensions(self):
                return frozenset({".rs"})

            def analyze_file(self, path, content):
                return [], []

        reg = AnalyzerRegistry()
        reg.register(RustAnalyzer())
        assert reg.get("/src/main.rs") is not None

    def test_analyze_file_convenience(self):
        reg = AnalyzerRegistry()
        result = reg.analyze_file("/src/mod.py", "def foo(): pass\n")
        assert result is not None
        chunks = result[0]
        assert len(chunks) == 1

    def test_analyze_file_unsupported(self):
        reg = AnalyzerRegistry()
        result = reg.analyze_file("/data.csv", "a,b,c\n")
        assert result is None

    def test_case_insensitive(self):
        reg = AnalyzerRegistry()
        assert reg.get("/README.PY") is not None


# ===================================================================
# TestGetAnalyzer
# ===================================================================


class TestGetAnalyzer:
    def test_returns_python(self):
        analyzer = get_analyzer("/src/mod.py")
        assert analyzer is not None
        assert isinstance(analyzer, PythonAnalyzer)

    def test_returns_none_for_unknown(self):
        assert get_analyzer("/data.xml") is None


# ===================================================================
# Integration tests — require cloned repos
# ===================================================================


@pytest.mark.integration
class TestFlaskIntegration:
    @pytest.fixture(autouse=True)
    def _check_repo(self):
        repo = FIXTURES_DIR / "flask"
        if not repo.exists():
            pytest.skip("Flask repo not cloned")

    def _read_flask_file(self, relpath):
        return (FIXTURES_DIR / "flask" / relpath).read_text()

    def test_analyze_flask_app(self):
        analyzer = PythonAnalyzer()
        content = self._read_flask_file("src/flask/app.py")
        chunks = analyzer.analyze_file("/src/flask/app.py", content)[0]
        assert len(chunks) > 0

    def test_contains_edges(self):
        analyzer = PythonAnalyzer()
        content = self._read_flask_file("src/flask/app.py")
        edges = analyzer.analyze_file("/src/flask/app.py", content)[1]
        contains = [e for e in edges if e.edge_type == "contains"]
        assert len(contains) > 0

    def test_imports_edges(self):
        analyzer = PythonAnalyzer()
        content = self._read_flask_file("src/flask/app.py")
        edges = analyzer.analyze_file("/src/flask/app.py", content)[1]
        imports = [e for e in edges if e.edge_type == "imports"]
        assert len(imports) > 0

    def test_flask_class_exists(self):
        analyzer = PythonAnalyzer()
        content = self._read_flask_file("src/flask/app.py")
        chunks = analyzer.analyze_file("/src/flask/app.py", content)[0]
        names = [c.name for c in chunks]
        assert "Flask" in names


@pytest.mark.integration
class TestExpressIntegration:
    @pytest.fixture(autouse=True)
    def _check_repo(self):
        repo = FIXTURES_DIR / "express"
        if not repo.exists():
            pytest.skip("Express repo not cloned")

    def test_analyze_express(self):
        from grover.graph.analyzers.javascript import JavaScriptAnalyzer

        # Express main entry
        content = (FIXTURES_DIR / "express" / "lib" / "express.js").read_text()
        analyzer = JavaScriptAnalyzer()
        chunks, edges = analyzer.analyze_file("/lib/express.js", content)
        # Should find at least some structure
        assert len(chunks) > 0 or len(edges) > 0


@pytest.mark.integration
class TestChiIntegration:
    @pytest.fixture(autouse=True)
    def _check_repo(self):
        repo = FIXTURES_DIR / "chi"
        if not repo.exists():
            pytest.skip("chi repo not cloned")

    def test_analyze_chi_mux(self):
        from grover.graph.analyzers.go import GoAnalyzer

        content = (FIXTURES_DIR / "chi" / "mux.go").read_text()
        analyzer = GoAnalyzer()
        chunks = analyzer.analyze_file("/mux.go", content)[0]
        assert len(chunks) > 0

    def test_method_scoping(self):
        from grover.graph.analyzers.go import GoAnalyzer

        content = (FIXTURES_DIR / "chi" / "mux.go").read_text()
        analyzer = GoAnalyzer()
        chunks = analyzer.analyze_file("/mux.go", content)[0]
        # Should have scoped method names like "Mux.ServeHTTP"
        scoped = [c.name for c in chunks if "." in c.name]
        assert len(scoped) > 0
