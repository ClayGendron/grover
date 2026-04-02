"""Tests for DatabricksVectorStore — mocks the Databricks SDK entirely."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestDatabricksImportGuard:
    def test_raises_when_sdk_missing(self, monkeypatch: pytest.MonkeyPatch):
        import grover.databricks_store as mod

        monkeypatch.setattr(mod, "_HAS_DATABRICKS", False)
        with pytest.raises(ImportError, match="databricks-vectorsearch is required"):
            mod.DatabricksVectorStore(index_name="idx", endpoint_name="ep")


class TestDatabricksQuery:
    async def test_user_id_filter(self):
        from grover.databricks_store import DatabricksVectorStore

        store = DatabricksVectorStore(index_name="idx", endpoint_name="ep")
        mock_index = MagicMock()
        mock_index.similarity_search.return_value = {
            "manifest": {"columns": [{"name": "id"}, {"name": "score"}]},
            "result": {"data_array": [["path/a.py", 0.9]]},
        }
        store._index = mock_index

        hits = await store.query([0.1, 0.2], k=5, user_id="alice")
        call_kwargs = mock_index.similarity_search.call_args[1]
        assert call_kwargs["filters"] == {"owner_id": "alice"}
        assert len(hits) == 1


class TestDatabricksConnect:
    async def test_connect_with_host_and_token(self):
        from grover.databricks_store import DatabricksVectorStore

        store = DatabricksVectorStore(
            index_name="idx", endpoint_name="ep", host="https://host", token="tok123",
        )

        mock_client_cls = MagicMock()
        mock_index = MagicMock()
        mock_client_cls.return_value = MagicMock()

        with (
            patch("grover.databricks_store.VectorSearchClient", mock_client_cls),
            patch("grover.databricks_store.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = AsyncMock(return_value=mock_index)
            await store.connect()

        mock_client_cls.assert_called_once_with(
            workspace_url="https://host",
            personal_access_token="tok123",
        )
