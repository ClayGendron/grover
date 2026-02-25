"""Tests for FileSearchResult.rebase() and remap_paths()."""

from __future__ import annotations

from grover.results import Evidence, FileSearchResult
from grover.search.results import (
    GlobEvidence,
    GlobResult,
    GrepEvidence,
    GrepResult,
    LineMatch,
    ListDirEvidence,
    ListDirResult,
    TreeEvidence,
    TreeResult,
)

# =====================================================================
# Helpers
# =====================================================================


def _glob(paths: dict[str, bool], *, pattern: str = "*.py") -> GlobResult:
    """Build a GlobResult from {path: is_directory} mapping."""
    entries: dict[str, list[Evidence]] = {}
    for p, is_d in paths.items():
        entries[p] = [GlobEvidence(strategy="glob", path=p, is_directory=is_d, size_bytes=100)]
    return GlobResult(
        success=True,
        message=f"Found {len(paths)} match(es)",
        _entries=entries,
        pattern=pattern,
    )


def _grep(
    matches: dict[str, list[tuple[int, str]]],
    *,
    pattern: str = "login",
) -> GrepResult:
    """Build a GrepResult from {path: [(line_no, content), ...]} mapping."""
    entries: dict[str, list[Evidence]] = {}
    for p, lms in matches.items():
        line_matches = tuple(LineMatch(line_number=ln, line_content=lc) for ln, lc in lms)
        entries[p] = [GrepEvidence(strategy="grep", path=p, line_matches=line_matches)]
    return GrepResult(
        success=True,
        message=f"Found matches in {len(matches)} file(s)",
        _entries=entries,
        pattern=pattern,
        files_searched=10,
        files_matched=len(matches),
    )


def _tree(paths: dict[str, tuple[int, bool]]) -> TreeResult:
    """Build a TreeResult from {path: (depth, is_directory)} mapping."""
    entries: dict[str, list[Evidence]] = {}
    for p, (depth, is_d) in paths.items():
        entries[p] = [TreeEvidence(strategy="tree", path=p, depth=depth, is_directory=is_d)]
    return TreeResult(
        success=True,
        message=f"Found {len(paths)} entries",
        _entries=entries,
    )


def _list_dir(paths: dict[str, bool]) -> ListDirResult:
    """Build a ListDirResult from {path: is_directory} mapping."""
    entries: dict[str, list[Evidence]] = {}
    for p, is_d in paths.items():
        entries[p] = [ListDirEvidence(strategy="list_dir", path=p, is_directory=is_d)]
    return ListDirResult(
        success=True,
        message=f"Found {len(paths)} entries",
        _entries=entries,
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

    def test_glob_rebase_updates_evidence_paths(self) -> None:
        result = _glob({"/src/main.py": False})
        rebased = result.rebase("/mount")

        evs = rebased.explain("/mount/src/main.py")
        assert len(evs) == 1
        assert evs[0].path == "/mount/src/main.py"
        assert isinstance(evs[0], GlobEvidence)
        assert evs[0].is_directory is False
        assert evs[0].size_bytes == 100

    def test_grep_rebase_prefixes_paths(self) -> None:
        result = _grep({"/auth.py": [(10, "def login():")], "/db.py": [(5, "pool")]})
        rebased = result.rebase("/project")

        assert "/project/auth.py" in rebased
        assert "/project/db.py" in rebased
        assert "/auth.py" not in rebased

    def test_grep_rebase_preserves_pattern_and_stats(self) -> None:
        result = _grep({"/auth.py": [(10, "def login():")]}, pattern="login")
        rebased = result.rebase("/project")

        assert rebased.pattern == "login"
        assert rebased.files_searched == 10
        assert rebased.files_matched == 1

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
        assert rebased.total_files == 1
        assert rebased.total_dirs == 1

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

    def test_rebase_preserves_subclass_type(self) -> None:
        glob_result = _glob({"/a.py": False})
        assert isinstance(glob_result.rebase("/m"), GlobResult)

        grep_result = _grep({"/a.py": [(1, "x")]})
        assert isinstance(grep_result.rebase("/m"), GrepResult)

        tree_result = _tree({"/a": (1, True)})
        assert isinstance(tree_result.rebase("/m"), TreeResult)

        list_result = _list_dir({"/a": True})
        assert isinstance(list_result.rebase("/m"), ListDirResult)

    def test_rebase_root_path(self) -> None:
        """Root path '/' gets replaced with just the prefix."""
        entries: dict[str, list[Evidence]] = {"/": [Evidence(strategy="test", path="/")]}
        result = FileSearchResult(success=True, message="test", _entries=entries)
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


# =====================================================================
# remap_paths() tests
# =====================================================================


class TestRemapPaths:
    def test_remap_transforms_paths(self) -> None:
        result = _glob({"/user1/docs/a.txt": False, "/user1/docs/b.txt": False})
        remapped = result.remap_paths(lambda p: p.replace("/user1/", "/@shared/alice/"))

        assert "/@shared/alice/docs/a.txt" in remapped
        assert "/@shared/alice/docs/b.txt" in remapped

    def test_remap_updates_evidence_paths(self) -> None:
        result = _glob({"/src/a.py": False})
        remapped = result.remap_paths(lambda p: "/new" + p)

        evs = remapped.explain("/new/src/a.py")
        assert len(evs) == 1
        assert evs[0].path == "/new/src/a.py"

    def test_remap_preserves_subclass_type(self) -> None:
        result = _grep({"/a.py": [(1, "x")]})
        remapped = result.remap_paths(lambda p: "/prefix" + p)

        assert isinstance(remapped, GrepResult)
        assert remapped.pattern == "login"

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
