"""Tests for internal Evidence types."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from grover.models.internal.evidence import (
    Evidence,
    GlobEvidence,
    GraphCentralityEvidence,
    GraphRelationshipEvidence,
    GrepEvidence,
    HybridEvidence,
    LexicalEvidence,
    LineMatch,
    ListDirEvidence,
    ShareEvidence,
    TrashEvidence,
    TreeEvidence,
    VectorEvidence,
    VersionEvidence,
)
from grover.models.internal.ref import File


class TestEvidence:
    def test_base_evidence(self):
        e = Evidence(operation="glob", score=0.5)
        assert e.operation == "glob"
        assert e.score == 0.5
        assert e.query_args == {}

    def test_frozen(self):
        e = Evidence(operation="glob")
        with pytest.raises(dataclasses.FrozenInstanceError):
            e.operation = "grep"  # type: ignore[misc]

    def test_with_query_args(self):
        e = Evidence(operation="search", query_args={"k": 10})
        assert e.query_args == {"k": 10}


class TestGlobEvidence:
    def test_defaults(self):
        e = GlobEvidence(operation="glob")
        assert e.is_directory is False
        assert e.size_bytes is None
        assert e.mime_type is None

    def test_with_metadata(self):
        e = GlobEvidence(operation="glob", is_directory=True, size_bytes=1024)
        assert e.is_directory is True
        assert e.size_bytes == 1024

    def test_inherits_evidence(self):
        e = GlobEvidence(operation="glob", score=0.9)
        assert isinstance(e, Evidence)
        assert e.score == 0.9


class TestGrepEvidence:
    def test_defaults(self):
        e = GrepEvidence(operation="grep")
        assert e.line_matches == ()

    def test_with_matches(self):
        lm = LineMatch(line_number=10, line_content="hello world")
        e = GrepEvidence(operation="grep", line_matches=(lm,))
        assert len(e.line_matches) == 1
        assert e.line_matches[0].line_number == 10


class TestLineMatch:
    def test_construction(self):
        lm = LineMatch(line_number=5, line_content="x = 1")
        assert lm.line_number == 5
        assert lm.line_content == "x = 1"
        assert lm.context_before == ()
        assert lm.context_after == ()

    def test_with_context(self):
        lm = LineMatch(
            line_number=5,
            line_content="x = 1",
            context_before=("# comment",),
            context_after=("y = 2",),
        )
        assert lm.context_before == ("# comment",)
        assert lm.context_after == ("y = 2",)


class TestTreeEvidence:
    def test_defaults(self):
        e = TreeEvidence(operation="tree")
        assert e.depth == 0
        assert e.is_directory is False


class TestListDirEvidence:
    def test_defaults(self):
        e = ListDirEvidence(operation="list_dir")
        assert e.is_directory is False
        assert e.size_bytes is None


class TestTrashEvidence:
    def test_defaults(self):
        e = TrashEvidence(operation="trash")
        assert e.deleted_at is None
        assert e.original_path == ""

    def test_with_metadata(self):
        now = datetime.now(UTC)
        e = TrashEvidence(operation="trash", deleted_at=now, original_path="/old.py")
        assert e.deleted_at == now
        assert e.original_path == "/old.py"


class TestVectorEvidence:
    def test_defaults(self):
        e = VectorEvidence(operation="vector_search")
        assert e.snippet == ""

    def test_with_snippet(self):
        e = VectorEvidence(operation="vector_search", score=0.95, snippet="relevant code")
        assert e.snippet == "relevant code"
        assert e.score == 0.95


class TestLexicalEvidence:
    def test_defaults(self):
        e = LexicalEvidence(operation="lexical_search")
        assert e.snippet == ""


class TestHybridEvidence:
    def test_defaults(self):
        e = HybridEvidence(operation="hybrid_search")
        assert e.snippet == ""


class TestGraphRelationshipEvidence:
    def test_defaults(self):
        e = GraphRelationshipEvidence(operation="predecessors")
        assert e.paths == []

    def test_with_paths(self):
        e = GraphRelationshipEvidence(
            operation="predecessors",
            score=0.8,
            paths=["/a.py", "/b.py"],
        )
        assert e.operation == "predecessors"
        assert e.score == 0.8
        assert e.paths == ["/a.py", "/b.py"]


class TestGraphCentralityEvidence:
    def test_defaults(self):
        e = GraphCentralityEvidence(operation="pagerank")
        assert e.scores == {}

    def test_with_scores(self):
        e = GraphCentralityEvidence(
            operation="pagerank",
            score=0.75,
            scores={"pagerank": 0.75},
        )
        assert e.operation == "pagerank"
        assert e.score == 0.75
        assert e.scores == {"pagerank": 0.75}


class TestVersionEvidence:
    def test_defaults(self):
        e = VersionEvidence(operation="list_versions")
        assert e.version == 0
        assert e.content_hash == ""
        assert e.size_bytes == 0
        assert e.created_at is None
        assert e.created_by is None

    def test_with_metadata(self):
        now = datetime.now(UTC)
        e = VersionEvidence(
            operation="list_versions",
            version=3,
            content_hash="abc123",
            size_bytes=1024,
            created_at=now,
            created_by="agent",
        )
        assert e.version == 3
        assert e.content_hash == "abc123"
        assert e.created_by == "agent"


class TestShareEvidence:
    def test_defaults(self):
        e = ShareEvidence(operation="list_shares")
        assert e.grantee_id == ""
        assert e.permission == ""
        assert e.granted_by == ""
        assert e.expires_at is None


class TestEvidenceOnRefs:
    def test_evidence_attaches_to_file(self):
        e = GlobEvidence(operation="glob", is_directory=False)
        f = File(path="/a.py", evidence=[e])
        assert len(f.evidence) == 1
        assert isinstance(f.evidence[0], GlobEvidence)

    def test_multiple_evidence_types(self):
        e1 = GlobEvidence(operation="glob")
        e2 = VectorEvidence(operation="vector_search", score=0.9)
        f = File(path="/a.py", evidence=[e1, e2])
        assert len(f.evidence) == 2
