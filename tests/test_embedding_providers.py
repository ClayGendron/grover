"""Tests for embedding providers (Phase 2)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from grover.search.protocols import EmbeddingProvider

# ==================================================================
# OpenAI provider
# ==================================================================


class TestOpenAIEmbedding:
    def _make_provider(self, **kwargs):
        from grover.search.providers.openai import OpenAIEmbedding

        return OpenAIEmbedding(api_key="sk-test-key", **kwargs)

    def _mock_response(self, vectors: list[list[float]]):
        """Build a mock CreateEmbeddingResponse."""
        mock_resp = MagicMock()
        mock_data = []
        for i, vec in enumerate(vectors):
            item = MagicMock()
            item.embedding = vec
            item.index = i
            mock_data.append(item)
        mock_resp.data = mock_data
        return mock_resp

    @pytest.mark.asyncio
    async def test_embed_single_text(self):
        provider = self._make_provider()
        expected = [0.1, 0.2, 0.3]
        provider._client.embeddings.create = AsyncMock(
            return_value=self._mock_response([expected])
        )

        result = await provider.embed("hello")

        assert result == expected
        provider._client.embeddings.create.assert_called_once()
        call_kwargs = provider._client.embeddings.create.call_args[1]
        assert call_kwargs["input"] == ["hello"]
        assert call_kwargs["model"] == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        provider = self._make_provider()
        vecs = [[0.1, 0.2], [0.3, 0.4]]
        provider._client.embeddings.create = AsyncMock(
            return_value=self._mock_response(vecs)
        )

        result = await provider.embed_batch(["hello", "world"])

        assert result == vecs

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self):
        provider = self._make_provider()
        result = await provider.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_batch_chunking(self):
        provider = self._make_provider(batch_size=2)

        def make_response(texts):
            vecs = [[float(i)] for i in range(len(texts))]
            return self._mock_response(vecs)

        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            resp = make_response(kwargs["input"])
            # Fix indices for the batch
            for i, item in enumerate(resp.data):
                item.index = i
                item.embedding = [float(call_count * 10 + i)]
            call_count += 1
            return resp

        provider._client.embeddings.create = mock_create

        result = await provider.embed_batch(["a", "b", "c", "d", "e"])

        assert call_count == 3  # ceil(5/2) = 3 calls
        assert len(result) == 5

    def test_dimensions_from_config(self):
        provider = self._make_provider(dimensions=256)
        assert provider.dimensions == 256

    def test_dimensions_default_small(self):
        provider = self._make_provider()
        assert provider.dimensions == 1536

    def test_dimensions_default_large(self):
        provider = self._make_provider(model="text-embedding-3-large")
        assert provider.dimensions == 3072

    def test_dimensions_default_ada(self):
        provider = self._make_provider(model="text-embedding-ada-002")
        assert provider.dimensions == 1536

    def test_dimensions_unknown_model_raises(self):
        provider = self._make_provider(model="custom-model")
        with pytest.raises(ValueError, match="Unknown default dimensions"):
            _ = provider.dimensions

    def test_model_name(self):
        provider = self._make_provider()
        assert provider.model_name == "text-embedding-3-small"

    def test_custom_model_name(self):
        provider = self._make_provider(model="text-embedding-3-large")
        assert provider.model_name == "text-embedding-3-large"

    @pytest.mark.asyncio
    async def test_close(self):
        provider = self._make_provider()
        provider._client.close = AsyncMock()
        await provider.close()
        provider._client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_dimensions_passed_to_api(self):
        provider = self._make_provider(dimensions=256)
        provider._client.embeddings.create = AsyncMock(
            return_value=self._mock_response([[0.1] * 256])
        )

        await provider.embed("test")

        call_kwargs = provider._client.embeddings.create.call_args[1]
        assert call_kwargs["dimensions"] == 256

    @pytest.mark.asyncio
    async def test_dimensions_not_passed_when_none(self):
        provider = self._make_provider()
        provider._client.embeddings.create = AsyncMock(
            return_value=self._mock_response([[0.1] * 1536])
        )

        await provider.embed("test")

        call_kwargs = provider._client.embeddings.create.call_args[1]
        assert "dimensions" not in call_kwargs

    def test_api_key_required(self):
        from grover.search.providers.openai import OpenAIEmbedding

        with patch.dict("os.environ", {}, clear=True):
            # Remove any OPENAI_API_KEY from env
            import os

            env_backup = os.environ.pop("OPENAI_API_KEY", None)
            try:
                with pytest.raises(ValueError, match="No OpenAI API key"):
                    OpenAIEmbedding()
            finally:
                if env_backup is not None:
                    os.environ["OPENAI_API_KEY"] = env_backup

    def test_api_key_from_env(self):
        from grover.search.providers.openai import OpenAIEmbedding

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-from-env"}):
            provider = OpenAIEmbedding()
            assert provider.model_name == "text-embedding-3-small"

    def test_isinstance_embedding_provider(self):
        provider = self._make_provider()
        assert isinstance(provider, EmbeddingProvider)

    @pytest.mark.asyncio
    async def test_response_sorted_by_index(self):
        """Verify vectors are returned in input order even if API returns out of order."""
        provider = self._make_provider()
        mock_resp = MagicMock()
        # Return items out of order
        item0 = MagicMock()
        item0.embedding = [0.0]
        item0.index = 1
        item1 = MagicMock()
        item1.embedding = [1.0]
        item1.index = 0
        mock_resp.data = [item0, item1]

        provider._client.embeddings.create = AsyncMock(return_value=mock_resp)

        result = await provider.embed_batch(["first", "second"])

        # Should be sorted by index: index 0 first, index 1 second
        assert result == [[1.0], [0.0]]


# ==================================================================
# SentenceTransformer provider
# ==================================================================


class TestSentenceTransformerEmbedding:
    def test_model_name_default(self):
        from grover.search.providers.sentence_transformers import (
            SentenceTransformerEmbedding,
        )

        p = SentenceTransformerEmbedding.__new__(SentenceTransformerEmbedding)
        p._model_name = "all-MiniLM-L6-v2"
        p._model = None
        assert p.model_name == "all-MiniLM-L6-v2"

    def test_model_name_custom(self):
        from grover.search.providers.sentence_transformers import (
            SentenceTransformerEmbedding,
        )

        p = SentenceTransformerEmbedding.__new__(SentenceTransformerEmbedding)
        p._model_name = "custom-model"
        p._model = None
        assert p.model_name == "custom-model"

    def test_isinstance_embedding_provider(self):
        from grover.search.providers.sentence_transformers import (
            SentenceTransformerEmbedding,
        )

        p = SentenceTransformerEmbedding.__new__(SentenceTransformerEmbedding)
        p._model_name = "test"
        p._model = None
        assert isinstance(p, EmbeddingProvider)

    @pytest.mark.asyncio
    async def test_embed_async_uses_thread_pool(self):
        import numpy as np

        from grover.search.providers.sentence_transformers import (
            SentenceTransformerEmbedding,
        )

        p = SentenceTransformerEmbedding.__new__(SentenceTransformerEmbedding)
        p._model_name = "test"
        p._model = MagicMock()
        p._model.encode = MagicMock(return_value=np.array([[0.1, 0.2]]))

        # embed_sync should work synchronously
        vec = p.embed_sync("hello")
        assert isinstance(vec, list)
        assert vec == pytest.approx([0.1, 0.2])

        # embed should be async (returns a coroutine)
        result = p.embed("hello")
        assert hasattr(result, "__await__")
        vec = await result
        assert isinstance(vec, list)

    @pytest.mark.asyncio
    async def test_embed_batch_async_uses_thread_pool(self):
        import numpy as np

        from grover.search.providers.sentence_transformers import (
            SentenceTransformerEmbedding,
        )

        p = SentenceTransformerEmbedding.__new__(SentenceTransformerEmbedding)
        p._model_name = "test"
        p._model = MagicMock()
        p._model.encode = MagicMock(
            return_value=np.array([[0.1, 0.2], [0.3, 0.4]])
        )

        # embed_batch_sync should work synchronously
        vecs = p.embed_batch_sync(["hello", "world"])
        assert len(vecs) == 2

        # embed_batch should be async
        result = p.embed_batch(["hello", "world"])
        assert hasattr(result, "__await__")
        vecs = await result
        assert len(vecs) == 2


# ==================================================================
# LangChain adapter
# ==================================================================


class TestLangChainEmbedding:
    def _make_mock_embeddings(self, *, vec=None, model_name=None):
        """Create a mock LangChain Embeddings instance."""
        from langchain_core.embeddings import Embeddings

        mock = MagicMock(spec=Embeddings)
        vec = vec or [0.1, 0.2, 0.3]
        mock.aembed_query = AsyncMock(return_value=vec)
        mock.aembed_documents = AsyncMock(return_value=[vec, vec])
        mock.embed_query = MagicMock(return_value=vec)
        if model_name:
            mock.model = model_name
        return mock

    def _make_provider(self, *, mock_embeddings=None, **kwargs):
        from grover.search.providers.langchain import LangChainEmbedding

        embeddings = mock_embeddings or self._make_mock_embeddings()
        return LangChainEmbedding(embeddings, **kwargs)

    @pytest.mark.asyncio
    async def test_embed_delegates_to_aembed_query(self):
        mock = self._make_mock_embeddings()
        provider = self._make_provider(mock_embeddings=mock)

        result = await provider.embed("hello")

        assert result == [0.1, 0.2, 0.3]
        mock.aembed_query.assert_called_once_with("hello")

    @pytest.mark.asyncio
    async def test_embed_batch_delegates_to_aembed_documents(self):
        mock = self._make_mock_embeddings()
        provider = self._make_provider(mock_embeddings=mock)

        result = await provider.embed_batch(["hello", "world"])

        assert len(result) == 2
        mock.aembed_documents.assert_called_once_with(["hello", "world"])

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self):
        mock = self._make_mock_embeddings()
        provider = self._make_provider(mock_embeddings=mock)

        result = await provider.embed_batch([])

        assert result == []
        mock.aembed_documents.assert_not_called()

    def test_dimensions_from_constructor(self):
        provider = self._make_provider(dimensions=384)
        assert provider.dimensions == 384

    def test_dimensions_probed(self):
        mock = self._make_mock_embeddings(vec=[0.1, 0.2, 0.3, 0.4])
        provider = self._make_provider(mock_embeddings=mock)

        dim = provider.dimensions

        assert dim == 4
        mock.embed_query.assert_called_once_with("dimension probe")

    def test_dimensions_probed_cached(self):
        mock = self._make_mock_embeddings(vec=[0.1, 0.2, 0.3])
        provider = self._make_provider(mock_embeddings=mock)

        _ = provider.dimensions
        _ = provider.dimensions  # Second call should use cache

        mock.embed_query.assert_called_once()  # Only probed once

    def test_model_name_from_constructor(self):
        provider = self._make_provider(model_name="my-custom-model")
        assert provider.model_name == "my-custom-model"

    def test_model_name_discovered_from_model_attr(self):
        mock = self._make_mock_embeddings()
        mock.model = "gpt-embeddings"
        provider = self._make_provider(mock_embeddings=mock)
        assert provider.model_name == "gpt-embeddings"

    def test_model_name_discovered_from_model_name_attr(self):
        mock = self._make_mock_embeddings()
        # Remove 'model' attr, add 'model_name'
        del mock.model
        mock.model_name = "hf-embeddings"
        provider = self._make_provider(mock_embeddings=mock)
        assert provider.model_name == "hf-embeddings"

    def test_model_name_falls_back_to_class_name(self):
        mock = self._make_mock_embeddings()
        # Remove model name attrs
        del mock.model
        provider = self._make_provider(mock_embeddings=mock)
        assert provider.model_name == "MagicMock"

    def test_rejects_non_embeddings_instance(self):
        from grover.search.providers.langchain import LangChainEmbedding

        with pytest.raises(TypeError, match="must be a langchain_core"):
            LangChainEmbedding("not an embeddings object")

    def test_isinstance_embedding_provider(self):
        provider = self._make_provider()
        assert isinstance(provider, EmbeddingProvider)
