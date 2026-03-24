"""Tests for FileSearchResult.rebase() and remap_paths()."""

from __future__ import annotations

from grover.models.internal.evidence import (
    Evidence,
    GlobEvidence,
    GrepEvidence,
    LineMatch,
    ListDirEvidence,
    TreeEvidence,
)
from grover.models.internal.ref import File, FileConnection
from grover.models.internal.results import FileSearchResult

# =====================================================================
# Helpers
# =====================================================================


def _glob(paths: dict[str, bool], *, pattern: str = "*.py") -> FileSearchResult:
    """Build a FileSearchResult from {path: is_directory} mapping."""
    files = [
        File(
            path=p,
            evidence=[GlobEvidence(operation="glob", is_directory=is_d, size_bytes=100)],
        )
        for p, is_d in paths.items()
    ]
    return FileSearchResult(
        success=True,
        message=f"Found {len(paths)} match(es)",
        files=files,
    )


def _grep(
    matches: dict[str, list[tuple[int, str]]],
    *,
    pattern: str = "login",
) -> FileSearchResult:
    """Build a FileSearchResult from {path: [(line_no, content), ...]} mapping."""
    files = []
    for p, lms in matches.items():
        line_matches = tuple(LineMatch(line_number=ln, line_content=lc) for ln, lc in lms)
        files.append(
            File(
                path=p,
                evidence=[GrepEvidence(operation="grep", line_matches=line_matches)],
            )
        )
    return FileSearchResult(
        success=True,
        message=f"Found matches in {len(matches)} file(s)",
        files=files,
    )


def _tree(paths: dict[str, tuple[int, bool]]) -> FileSearchResult:
    """Build a FileSearchResult from {path: (depth, is_directory)} mapping."""
    files = [
        File(
            path=p,
            evidence=[TreeEvidence(operation="tree", depth=depth, is_directory=is_d)],
        )
        for p, (depth, is_d) in paths.items()
    ]
    return FileSearchResult(
        success=True,
        message=f"Found {len(paths)} entries",
        files=files,
    )


def _list_dir(paths: dict[str, bool]) -> FileSearchResult:
    """Build a FileSearchResult from {path: is_directory} mapping."""
    files = [
        File(
            path=p,
            evidence=[ListDirEvidence(operation="list_dir", is_directory=is_d)],
        )
        for p, is_d in paths.items()
    ]
    return FileSearchResult(
        success=True,
        message=f"Found {len(paths)} entries",
        files=files,
    )


# =====================================================================
# rebase() tests
# =====================================================================


class TestRebase:
    def test_glob_rebase_prefixes_paths(self) -> None:
        result = _glob({"/src/main.py": False, "/src/lib/": True})
        rebased = result.rebase("/mount")

        assert "/mount/src/main.py" in rebased
        assert "/mount/src/lib/" in rebased
        assert "/src/main.py" not in rebased

    def test_glob_rebase_preserves_evidence_fields(self) -> None:
        result = _glob({"/src/main.py": False})
        rebased = result.rebase("/mount")

        evs = rebased.explain("/mount/src/main.py")
        assert len(evs) == 1
        assert isinstance(evs[0], GlobEvidence)
        assert evs[0].is_directory is False
        assert evs[0].size_bytes == 100

    def test_grep_rebase_prefixes_paths(self) -> None:
        result = _grep({"/auth.py": [(10, "def login():")], "/db.py": [(5, "pool")]})
        rebased = result.rebase("/project")

        assert "/project/auth.py" in rebased
        assert "/project/db.py" in rebased
        assert "/auth.py" not in rebased

    def test_grep_rebase_preserves_line_matches(self) -> None:
        result = _grep({"/auth.py": [(10, "def login():"), (20, "  login()")]})
        rebased = result.rebase("/project")

        evs = rebased.explain("/project/auth.py")
        assert len(evs) == 1
        grep_ev = evs[0]
        assert isinstance(grep_ev, GrepEvidence)
        assert len(grep_ev.line_matches) == 2
        assert grep_ev.line_matches[0].line_number == 10
        assert grep_ev.line_matches[1].line_number == 20

    def test_tree_rebase_prefixes_paths(self) -> None:
        result = _tree({"/src": (1, True), "/src/main.py": (2, False)})
        rebased = result.rebase("/mount")

        assert "/mount/src" in rebased
        assert "/mount/src/main.py" in rebased

    def test_list_dir_rebase_prefixes_paths(self) -> None:
        result = _list_dir({"/README.md": False, "/src": True})
        rebased = result.rebase("/project")

        assert "/project/README.md" in rebased
        assert "/project/src" in rebased

    def test_rebase_empty_result(self) -> None:
        result = _glob({})
        rebased = result.rebase("/mount")

        assert len(rebased) == 0
        assert list(rebased) == []

    def test_rebase_preserves_type(self) -> None:
        result = _glob({"/a.py": False})
        rebased = result.rebase("/m")
        assert isinstance(rebased, FileSearchResult)

    def test_rebase_root_path(self) -> None:
        """Root path '/' gets replaced with just the prefix."""
        result = FileSearchResult(
            success=True,
            message="test",
            files=[File(path="/", evidence=[Evidence(operation="test")])],
        )
        rebased = result.rebase("/mount")

        assert "/mount" in rebased
        assert "/" not in rebased

    def test_rebase_does_not_mutate_original(self) -> None:
        result = _glob({"/a.py": False, "/b.py": False})
        _ = result.rebase("/mount")

        assert "/a.py" in result
        assert "/mount/a.py" not in result

    def test_rebase_then_set_algebra(self) -> None:
        """rebase + union works correctly across mounts."""
        r1 = _glob({"/a.py": False}).rebase("/mount1")
        r2 = _glob({"/b.py": False, "/a.py": False}).rebase("/mount2")
        combined = r1 | r2

        assert "/mount1/a.py" in combined
        assert "/mount2/b.py" in combined
        assert "/mount2/a.py" in combined
        assert len(combined) == 3

    def test_rebase_double_prefix(self) -> None:
        """Rebase can be applied twice (stacks prefixes)."""
        result = _glob({"/a.py": False})
        rebased = result.rebase("/inner").rebase("/outer")

        assert "/outer/inner/a.py" in rebased
        assert len(rebased) == 1

    def test_rebase_transforms_connections(self) -> None:
        conn = FileConnection(
            path="/a.py[imports]/b.py",
            source_path="/a.py",
            target_path="/b.py",
            type="imports",
            evidence=[Evidence(operation="graph")],
        )
        r = FileSearchResult(success=True, message="ok", connections=[conn])
        rebased = r.rebase("/mount")
        assert len(rebased.connections) == 1
        rc = rebased.connections[0]
        assert rc.source_path == "/mount/a.py"
        assert rc.target_path == "/mount/b.py"


# =====================================================================
# remap_paths() tests
# =====================================================================


class TestRemapPaths:
    def test_remap_transforms_paths(self) -> None:
        result = _glob({"/user1/docs/a.txt": False, "/user1/docs/b.txt": False})
        remapped = result.remap_paths(lambda p: p.replace("/user1/", "/@shared/alice/"))

        assert "/@shared/alice/docs/a.txt" in remapped
        assert "/@shared/alice/docs/b.txt" in remapped

    def test_remap_preserves_evidence_fields(self) -> None:
        result = _glob({"/src/a.py": False})
        remapped = result.remap_paths(lambda p: "/new" + p)

        evs = remapped.explain("/new/src/a.py")
        assert len(evs) == 1
        assert isinstance(evs[0], GlobEvidence)

    def test_remap_preserves_type(self) -> None:
        result = _grep({"/a.py": [(1, "x")]})
        remapped = result.remap_paths(lambda p: "/prefix" + p)
        assert isinstance(remapped, FileSearchResult)

    def test_remap_does_not_mutate_original(self) -> None:
        result = _list_dir({"/a": True, "/b": False})
        _ = result.remap_paths(lambda p: "/x" + p)

        assert "/a" in result
        assert "/x/a" not in result

    def test_remap_empty_result(self) -> None:
        result = _tree({})
        remapped = result.remap_paths(lambda p: "/x" + p)
        assert len(remapped) == 0

    def test_remap_identity(self) -> None:
        """Identity function produces equal result."""
        result = _glob({"/a.py": False, "/b.py": True})
        remapped = result.remap_paths(lambda p: p)

        assert set(remapped.paths) == set(result.paths)

    def test_remap_collapses_paths(self) -> None:
        """If fn maps two paths to the same string, entries merge."""
        result = _glob({"/a.py": False, "/b.py": False})
        remapped = result.remap_paths(lambda _p: "/merged.py")

        assert len(remapped) == 1
        assert "/merged.py" in remapped
        # Evidence from both should be present
        evs = remapped.explain("/merged.py")
        assert len(evs) == 2

    def test_remap_then_set_algebra(self) -> None:
        """Remap followed by intersection works."""
        r1 = _glob({"/user1/a.py": False, "/user1/b.py": False})
        r2 = _grep({"/shared/a.py": [(1, "x")]})

        r1_remapped = r1.remap_paths(lambda p: p.replace("/user1/", "/shared/"))
        intersection = r1_remapped & r2

        assert "/shared/a.py" in intersection
        assert "/shared/b.py" not in intersection

    def test_remap_paths_transforms_connections(self) -> None:
        conn = FileConnection(
            path="/a.py[imports]/b.py",
            source_path="/a.py",
            target_path="/b.py",
            type="imports",
            evidence=[Evidence(operation="graph")],
        )
        r = FileSearchResult(success=True, message="ok", connections=[conn])
        remapped = r.remap_paths(lambda p: "/new" + p)
        assert len(remapped.connections) == 1
        rc = remapped.connections[0]
        assert rc.source_path == "/new/a.py"
        assert rc.target_path == "/new/b.py"
