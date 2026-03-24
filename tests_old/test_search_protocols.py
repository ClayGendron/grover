"""Tests for search protocols, IndexConfig, and filter AST."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from grover.providers.embedding.protocol import EmbeddingProvider
from grover.providers.search.filters import (
    Comparison,
    FilterOp,
    LogicalGroup,
    LogicalOp,
    and_,
    compile_databricks,
    compile_dict,
    compile_pinecone,
    eq,
    exists,
    gt,
    gte,
    in_,
    lt,
    lte,
    ne,
    not_in,
    or_,
)
from grover.providers.search.protocol import IndexConfig, SearchProvider, parent_path_from_id

# ==================================================================
# IndexConfig
# ==================================================================


class TestIndexConfig:
    def test_construction(self):
        ic = IndexConfig(name="my-index", dimension=384)
        assert ic.name == "my-index"
        assert ic.dimension == 384
        assert ic.metric == "cosine"
        assert ic.cloud_config == {}

    def test_custom_metric(self):
        ic = IndexConfig(name="idx", dimension=768, metric="dotproduct")
        assert ic.metric == "dotproduct"

    def test_cloud_config(self):
        ic = IndexConfig(
            name="idx",
            dimension=384,
            cloud_config={"cloud": "aws", "region": "us-east-1"},
        )
        assert ic.cloud_config["cloud"] == "aws"

    def test_frozen(self):
        ic = IndexConfig(name="idx", dimension=384)
        with pytest.raises(FrozenInstanceError):
            ic.name = "other"  # type: ignore[misc]


# ==================================================================
# parent_path_from_id
# ==================================================================


class TestParentPathFromId:
    def test_chunk_id(self):
        assert parent_path_from_id("/a.py#login") == "/a.py"

    def test_plain_path(self):
        assert parent_path_from_id("/a.py") == "/a.py"

    def test_nested_hash(self):
        assert parent_path_from_id("/a.py#foo#bar") == "/a.py"


# ==================================================================
# Filter AST — construction via builder helpers
# ==================================================================


class TestFilterBuilders:
    def test_eq(self):
        f = eq("genre", "comedy")
        assert isinstance(f, Comparison)
        assert f.field == "genre"
        assert f.op == FilterOp.EQ
        assert f.value == "comedy"

    def test_ne(self):
        f = ne("genre", "horror")
        assert f.op == FilterOp.NE
        assert f.value == "horror"

    def test_gt(self):
        f = gt("year", 2000)
        assert f.op == FilterOp.GT
        assert f.value == 2000

    def test_gte(self):
        f = gte("rating", 4.5)
        assert f.op == FilterOp.GTE
        assert f.value == 4.5

    def test_lt(self):
        f = lt("price", 10)
        assert f.op == FilterOp.LT
        assert f.value == 10

    def test_lte(self):
        f = lte("price", 9.99)
        assert f.op == FilterOp.LTE
        assert f.value == 9.99

    def test_in(self):
        f = in_("color", ["red", "blue"])
        assert f.op == FilterOp.IN
        assert f.value == ["red", "blue"]

    def test_not_in(self):
        f = not_in("status", ["deleted", "archived"])
        assert f.op == FilterOp.NOT_IN
        assert f.value == ["deleted", "archived"]

    def test_exists_default(self):
        f = exists("thumbnail")
        assert f.op == FilterOp.EXISTS
        assert f.value is True

    def test_exists_false(self):
        f = exists("thumbnail", exists=False)
        assert f.value is False

    def test_and(self):
        f = and_(eq("a", 1), eq("b", 2))
        assert isinstance(f, LogicalGroup)
        assert f.op == LogicalOp.AND
        assert len(f.expressions) == 2

    def test_or(self):
        f = or_(eq("a", 1), eq("b", 2))
        assert f.op == LogicalOp.OR
        assert len(f.expressions) == 2

    def test_nested_and_or(self):
        f = and_(
            or_(eq("genre", "comedy"), eq("genre", "drama")),
            gt("year", 2000),
        )
        assert isinstance(f, LogicalGroup)
        assert f.op == LogicalOp.AND
        assert isinstance(f.expressions[0], LogicalGroup)
        assert f.expressions[0].op == LogicalOp.OR

    def test_comparison_is_frozen(self):
        f = eq("genre", "comedy")
        with pytest.raises(FrozenInstanceError):
            f.field = "year"  # type: ignore[misc]

    def test_logical_group_is_frozen(self):
        f = and_(eq("a", 1))
        with pytest.raises(FrozenInstanceError):
            f.op = LogicalOp.OR  # type: ignore[misc]


# ==================================================================
# Pinecone compiler
# ==================================================================


class TestCompilePinecone:
    def test_eq(self):
        assert compile_pinecone(eq("genre", "comedy")) == {"genre": {"$eq": "comedy"}}

    def test_ne(self):
        assert compile_pinecone(ne("genre", "horror")) == {"genre": {"$ne": "horror"}}

    def test_gt(self):
        assert compile_pinecone(gt("year", 2000)) == {"year": {"$gt": 2000}}

    def test_gte(self):
        assert compile_pinecone(gte("year", 2000)) == {"year": {"$gte": 2000}}

    def test_lt(self):
        assert compile_pinecone(lt("price", 10)) == {"price": {"$lt": 10}}

    def test_lte(self):
        assert compile_pinecone(lte("price", 10)) == {"price": {"$lte": 10}}

    def test_in(self):
        result = compile_pinecone(in_("color", ["red", "blue"]))
        assert result == {"color": {"$in": ["red", "blue"]}}

    def test_not_in(self):
        result = compile_pinecone(not_in("status", ["deleted"]))
        assert result == {"status": {"$nin": ["deleted"]}}

    def test_exists(self):
        assert compile_pinecone(exists("thumb")) == {"thumb": {"$exists": True}}

    def test_and(self):
        result = compile_pinecone(and_(eq("a", 1), gt("b", 2)))
        assert result == {"$and": [{"a": {"$eq": 1}}, {"b": {"$gt": 2}}]}

    def test_or(self):
        result = compile_pinecone(or_(eq("a", 1), eq("a", 2)))
        assert result == {"$or": [{"a": {"$eq": 1}}, {"a": {"$eq": 2}}]}

    def test_nested(self):
        result = compile_pinecone(
            and_(
                or_(eq("genre", "comedy"), eq("genre", "drama")),
                gt("year", 2000),
            )
        )
        assert result == {
            "$and": [
                {"$or": [{"genre": {"$eq": "comedy"}}, {"genre": {"$eq": "drama"}}]},
                {"year": {"$gt": 2000}},
            ]
        }


# ==================================================================
# Databricks compiler
# ==================================================================


class TestCompileDatabricks:
    def test_eq_string(self):
        assert compile_databricks(eq("genre", "comedy")) == "genre = 'comedy'"

    def test_eq_int(self):
        assert compile_databricks(eq("year", 2000)) == "year = 2000"

    def test_ne(self):
        assert compile_databricks(ne("genre", "horror")) == "genre != 'horror'"

    def test_gt(self):
        assert compile_databricks(gt("year", 2000)) == "year > 2000"

    def test_gte(self):
        assert compile_databricks(gte("year", 2000)) == "year >= 2000"

    def test_lt(self):
        assert compile_databricks(lt("price", 10)) == "price < 10"

    def test_lte(self):
        assert compile_databricks(lte("price", 9.99)) == "price <= 9.99"

    def test_in(self):
        result = compile_databricks(in_("color", ["red", "blue"]))
        assert result == "color IN ('red', 'blue')"

    def test_not_in(self):
        result = compile_databricks(not_in("status", ["deleted", "archived"]))
        assert result == "status NOT IN ('deleted', 'archived')"

    def test_exists_true(self):
        assert compile_databricks(exists("thumb")) == "thumb IS NOT NULL"

    def test_exists_false(self):
        assert compile_databricks(exists("thumb", exists=False)) == "thumb IS NULL"

    def test_and(self):
        result = compile_databricks(and_(eq("genre", "comedy"), gt("year", 2000)))
        assert result == "(genre = 'comedy' AND year > 2000)"

    def test_or(self):
        result = compile_databricks(or_(eq("a", 1), eq("b", 2)))
        assert result == "(a = 1 OR b = 2)"

    def test_nested(self):
        result = compile_databricks(
            and_(
                or_(eq("genre", "comedy"), eq("genre", "drama")),
                gt("year", 2000),
            )
        )
        assert result == "((genre = 'comedy' OR genre = 'drama') AND year > 2000)"

    def test_string_with_single_quote(self):
        result = compile_databricks(eq("title", "it's"))
        assert result == "title = 'it''s'"

    def test_bool_value(self):
        result = compile_databricks(eq("active", True))
        assert result == "active = TRUE"

    def test_in_with_ints(self):
        result = compile_databricks(in_("year", [2000, 2001, 2002]))
        assert result == "year IN (2000, 2001, 2002)"


# ==================================================================
# Dict compiler (local store)
# ==================================================================


class TestCompileDict:
    def test_eq(self):
        assert compile_dict(eq("genre", "comedy")) == {"genre": "comedy"}

    def test_and_of_eqs(self):
        result = compile_dict(and_(eq("genre", "comedy"), eq("year", 2000)))
        assert result == {"genre": "comedy", "year": 2000}

    def test_rejects_non_eq_operator(self):
        with pytest.raises(ValueError, match="only supports EQ"):
            compile_dict(gt("year", 2000))

    def test_rejects_or(self):
        with pytest.raises(ValueError, match="only supports AND"):
            compile_dict(or_(eq("a", 1), eq("b", 2)))

    def test_nested_and(self):
        result = compile_dict(and_(eq("a", 1), and_(eq("b", 2), eq("c", 3))))
        assert result == {"a": 1, "b": 2, "c": 3}


# ==================================================================
# Protocols — runtime checkability
# ==================================================================


class TestProtocolRuntimeChecks:
    def test_embedding_provider_is_runtime_checkable(self):
        class FakeEmbedding:
            async def embed(self, text: str) -> list[float]:
                return [0.0]

            async def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.0]]

            @property
            def dimensions(self) -> int:
                return 1

            @property
            def model_name(self) -> str:
                return "fake"

        assert isinstance(FakeEmbedding(), EmbeddingProvider)

    def test_non_provider_fails_check(self):
        class NotAProvider:
            pass

        assert not isinstance(NotAProvider(), EmbeddingProvider)

    def test_search_provider_is_runtime_checkable(self):
        class FakeSearch:
            async def connect(self):
                pass

            async def close(self):
                pass

            async def create_index(self, config):
                pass

            async def upsert(self, *, files):
                pass

            async def delete(self, *, files):
                pass

            async def vector_search(self, vector, *, k=10, candidates=None):
                pass

        assert isinstance(FakeSearch(), SearchProvider)

    def test_plain_object_fails_search_provider(self):
        assert not isinstance(object(), SearchProvider)
        assert not isinstance(object(), EmbeddingProvider)
