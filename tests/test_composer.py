"""Tests for multi-source composition (composer.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from sema_tree.composer import _reassign_ids, add_source, remove_source
from sema_tree.crawlers.base import CrawledPage
from sema_tree.llm import LLMProvider, LLMResponse
from sema_tree.models import SemaTree, IndexNode, RefType, Source, SourceType


class MockProvider(LLMProvider):
    async def generate(self, user_message, system="", max_tokens=256, temperature=0.0):
        return LLMResponse(text="DETAILED: Mock summary.\nNAVIGATIONAL: Mock nav")


def _make_pages(n: int) -> list[CrawledPage]:
    return [
        CrawledPage(
            url_or_path=f"https://example.com/page{i}",
            title=f"Page {i}",
            content=f"Content for page {i}",
            ref_type=RefType.url,
        )
        for i in range(n)
    ]


def _make_populated_index() -> SemaTree:
    """An index that already has one source."""
    return SemaTree(
        sources=[
            Source(id="first-src", type=SourceType.website, origin="https://first.com", page_count=2)
        ],
        root=IndexNode(
            id="0",
            title="Root",
            summary="First source summary",
            children=[
                IndexNode(
                    id="0.0",
                    title="First Source",
                    summary="First source summary",
                    source_id="first-src",
                    children=[
                        IndexNode(
                            id="0.0.0",
                            title="Page A",
                            summary="Page A",
                            ref="https://first.com/a",
                            ref_type=RefType.url,
                            source_id="first-src",
                        ),
                        IndexNode(
                            id="0.0.1",
                            title="Page B",
                            summary="Page B",
                            ref="https://first.com/b",
                            ref_type=RefType.url,
                            source_id="first-src",
                        ),
                    ],
                )
            ],
        ),
    )


class TestReassignIds:
    def test_reassigns_root(self):
        node = IndexNode(id="0.0", title="Node", summary="")
        _reassign_ids(node, "0.1")
        assert node.id == "0.1"

    def test_reassigns_children(self):
        root = IndexNode(
            id="0.0",
            title="Root",
            summary="",
            children=[
                IndexNode(id="0.0.0", title="A", summary=""),
                IndexNode(id="0.0.1", title="B", summary=""),
            ],
        )
        _reassign_ids(root, "0.2")
        assert root.id == "0.2"
        assert root.children[0].id == "0.2.0"
        assert root.children[1].id == "0.2.1"

    def test_reassigns_deeply_nested(self):
        root = IndexNode(
            id="0.0",
            title="Root",
            summary="",
            children=[
                IndexNode(
                    id="0.0.0",
                    title="A",
                    summary="",
                    children=[
                        IndexNode(id="0.0.0.0", title="A1", summary=""),
                    ],
                ),
            ],
        )
        _reassign_ids(root, "0.1")
        assert root.children[0].id == "0.1.0"
        assert root.children[0].children[0].id == "0.1.0.0"

    def test_preserves_node_count(self):
        root = IndexNode(
            id="0.0",
            title="Root",
            summary="",
            children=[
                IndexNode(id="0.0.0", title="A", summary=""),
                IndexNode(id="0.0.1", title="B", summary=""),
                IndexNode(id="0.0.2", title="C", summary=""),
            ],
        )
        before = root.count_nodes()
        _reassign_ids(root, "0.5")
        assert root.count_nodes() == before


class TestRemoveSource:
    @pytest.mark.asyncio
    async def test_removes_source_metadata(self):
        index = _make_populated_index()
        result = await remove_source(index, "first-src")
        assert len(result.sources) == 0

    @pytest.mark.asyncio
    async def test_removes_source_tree(self):
        index = _make_populated_index()
        result = await remove_source(index, "first-src")
        assert len(result.root.children) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_source_noop(self):
        index = _make_populated_index()
        result = await remove_source(index, "nonexistent-id")
        # Source list unchanged
        assert len(result.sources) == 1
        assert len(result.root.children) == 1

    @pytest.mark.asyncio
    async def test_remove_one_of_two_sources(self):
        index = _make_populated_index()
        # Add a second source tree manually
        index.sources.append(
            Source(id="second-src", type=SourceType.website, origin="https://second.com", page_count=1)
        )
        index.root.children.append(
            IndexNode(
                id="0.1",
                title="Second Source",
                summary="Second",
                source_id="second-src",
                children=[
                    IndexNode(
                        id="0.1.0",
                        title="Page C",
                        summary="Page C",
                        ref="https://second.com/c",
                        ref_type=RefType.url,
                        source_id="second-src",
                    )
                ],
            )
        )

        result = await remove_source(index, "first-src")
        assert len(result.sources) == 1
        assert result.sources[0].id == "second-src"
        assert len(result.root.children) == 1
        # Remaining child should be re-indexed to 0.0
        assert result.root.children[0].id == "0.0"


class TestAddSource:
    @pytest.mark.asyncio
    async def test_adds_source_to_empty_index(self):
        index = SemaTree()
        mock_pages = _make_pages(3)

        with patch("sema_tree.builder.WebCrawler") as MockWebCrawler:
            mock_crawler = AsyncMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_pages)
            MockWebCrawler.return_value = mock_crawler

            from sema_tree.builder import IndexBuilder
            builder = IndexBuilder(provider=MockProvider())
            result = await add_source(index, "https://example.com/docs", builder=builder)

        assert len(result.sources) == 1
        assert len(result.root.children) == 1

    @pytest.mark.asyncio
    async def test_adds_source_to_existing_index(self):
        index = _make_populated_index()
        mock_pages = _make_pages(2)

        with patch("sema_tree.builder.WebCrawler") as MockWebCrawler:
            mock_crawler = AsyncMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_pages)
            MockWebCrawler.return_value = mock_crawler

            from sema_tree.builder import IndexBuilder
            builder = IndexBuilder(provider=MockProvider())
            result = await add_source(index, "https://example.com/new", builder=builder)

        assert len(result.sources) == 2
        assert len(result.root.children) == 2

    @pytest.mark.asyncio
    async def test_add_source_updates_timestamp(self):
        index = _make_populated_index()
        original_ts = index.updated_at
        mock_pages = _make_pages(2)

        with patch("sema_tree.builder.WebCrawler") as MockWebCrawler:
            mock_crawler = AsyncMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_pages)
            MockWebCrawler.return_value = mock_crawler

            from sema_tree.builder import IndexBuilder
            builder = IndexBuilder(provider=MockProvider())
            result = await add_source(index, "https://example.com/new", builder=builder)

        assert result.updated_at >= original_ts
