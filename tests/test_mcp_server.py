"""Tests for the MCP server tools."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sema_tree.fs_store import FileSystemStore
from sema_tree.models import SemaTree, IndexNode, RefType, Source, SourceType


def _make_mixed_index(tmp_path: Path) -> tuple[FileSystemStore, SemaTree]:
    """Build and save an index with both URL and file ref nodes."""
    local_file = tmp_path / "sample.md"
    local_file.write_text("# Sample\n\nThis is a local doc.", encoding="utf-8")

    index = SemaTree(
        sources=[
            Source(id="web-src", type=SourceType.website, origin="https://example.com", page_count=1),
            Source(id="local-src", type=SourceType.local_folder, origin=str(tmp_path), page_count=1),
        ],
        root=IndexNode(
            id="0",
            title="Root",
            summary="Mixed index",
            children=[
                IndexNode(
                    id="0.0",
                    title="Web Doc",
                    summary="A web documentation page.",
                    nav_summary="Web doc summary",
                    ref="https://example.com/page",
                    ref_type=RefType.url,
                    source_id="web-src",
                ),
                IndexNode(
                    id="0.1",
                    title="Local Doc",
                    summary="A local file.",
                    nav_summary="Local file summary",
                    ref=str(local_file),
                    ref_type=RefType.file,
                    source_id="local-src",
                ),
            ],
        ),
    )

    store = FileSystemStore(tmp_path / "index")
    store.save(index)
    return store, index


class TestGetDetailsUrl:
    @pytest.mark.asyncio
    async def test_fetches_url_and_returns_content(self, tmp_path, monkeypatch):
        from sema_tree.server import mcp_server

        monkeypatch.chdir(tmp_path)
        store, _ = _make_mixed_index(tmp_path)
        mcp_server._store = store

        mock_resp = MagicMock()
        mock_resp.text = "# Web Page\n\nSome content."
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("sema_tree.server.mcp_server.httpx.AsyncClient", return_value=mock_client):
            result = await mcp_server.get_details("/web-doc.md")

        assert "Web Page" in result or "example.com" in result or "Live content" in result

    @pytest.mark.asyncio
    async def test_url_error_returns_error_message(self, tmp_path, monkeypatch):
        from sema_tree.server import mcp_server

        monkeypatch.chdir(tmp_path)
        store, _ = _make_mixed_index(tmp_path)
        mcp_server._store = store

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))

        with patch("sema_tree.server.mcp_server.httpx.AsyncClient", return_value=mock_client):
            result = await mcp_server.get_details("/web-doc.md")

        assert "Error" in result


class TestGetDetailsLocalFile:
    @pytest.mark.asyncio
    async def test_reads_local_file_no_http(self, tmp_path):
        """For ref_type=file, content is read from disk, not HTTP."""
        from sema_tree.server import mcp_server

        local_file = tmp_path / "mydoc.md"
        local_file.write_text("# My Local Doc\n\nLocal content here.", encoding="utf-8")

        index = SemaTree(
            sources=[Source(id="local", type=SourceType.local_folder, origin=str(tmp_path))],
            root=IndexNode(
                id="0",
                title="Root",
                children=[
                    IndexNode(
                        id="0.0",
                        title="My Doc",
                        summary="A local document.",
                        nav_summary="Local doc",
                        ref=str(local_file),
                        ref_type=RefType.file,
                        source_id="local",
                    )
                ],
            ),
        )
        store = FileSystemStore(tmp_path / "idx")
        store.save(index)
        mcp_server._store = store

        # Ensure httpx is never called
        with patch("sema_tree.server.mcp_server.httpx.AsyncClient") as mock_http:
            result = await mcp_server.get_details("/my-doc.md")
            mock_http.assert_not_called()

        assert "Local content here" in result

    @pytest.mark.asyncio
    async def test_local_file_not_found_returns_error(self, tmp_path):
        from sema_tree.server import mcp_server

        index = SemaTree(
            sources=[Source(id="local", type=SourceType.local_folder, origin=str(tmp_path))],
            root=IndexNode(
                id="0",
                title="Root",
                children=[
                    IndexNode(
                        id="0.0",
                        title="Missing",
                        summary="Gone.",
                        nav_summary="Gone",
                        ref="/nonexistent/path.md",
                        ref_type=RefType.file,
                        source_id="local",
                    )
                ],
            ),
        )
        store = FileSystemStore(tmp_path / "idx2")
        store.save(index)
        mcp_server._store = store

        result = await mcp_server.get_details("/missing.md")
        assert "Error" in result


class TestMixedIndexLsAndFind:
    def test_ls_returns_both_sources(self, tmp_path):
        from sema_tree.server import mcp_server

        store, _ = _make_mixed_index(tmp_path)
        mcp_server._store = store

        result = mcp_server.ls(path="/", depth=1)
        # Both nodes should appear
        assert "web-doc" in result.lower() or "web" in result.lower()
        assert "local-doc" in result.lower() or "local" in result.lower()

    def test_find_matches_across_sources(self, tmp_path):
        from sema_tree.server import mcp_server

        store, _ = _make_mixed_index(tmp_path)
        mcp_server._store = store

        result = mcp_server.find("doc")
        # Should find entries from both sources
        assert "doc" in result.lower()
