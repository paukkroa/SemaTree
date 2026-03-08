"""Tests for the IndexBuilder."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from sema_tree.builder import IndexBuilder, _detect_source_type, _slugify
from sema_tree.llm import LLMProvider, LLMResponse
from sema_tree.models import SourceType


class MockProvider(LLMProvider):
    """Mock LLM provider for testing."""

    async def generate(self, user_message, system="", max_tokens=256, temperature=0.0):
        return LLMResponse(text="Generated summary")


class TestSlugify:
    def test_url(self):
        result = _slugify("https://code.claude.com/docs/en")
        assert "code" in result
        assert "claude" in result

    def test_path(self):
        result = _slugify("/home/user/my docs")
        assert "home" in result
        assert "docs" in result

    def test_truncation(self):
        result = _slugify("a" * 100)
        assert len(result) <= 64


class TestDetectSourceType:
    def test_url(self):
        assert _detect_source_type("https://example.com") == SourceType.website
        assert _detect_source_type("http://example.com/docs") == SourceType.website

    def test_local_dir(self, tmp_path):
        assert _detect_source_type(str(tmp_path)) == SourceType.local_folder

    def test_unknown(self):
        with pytest.raises(ValueError):
            _detect_source_type("not_a_valid_source")


class TestIndexBuilder:
    @pytest.mark.asyncio
    async def test_build_website(self):
        from sema_tree.crawlers.base import CrawledPage
        from sema_tree.models import RefType

        mock_pages = [
            CrawledPage(
                url_or_path="https://example.com/page1",
                title="Page 1",
                content="Content 1",
                ref_type=RefType.url,
                metadata={"llms_txt_description": "Desc 1"},
            ),
            CrawledPage(
                url_or_path="https://example.com/page2",
                title="Page 2",
                content="Content 2",
                ref_type=RefType.url,
                metadata={"llms_txt_description": "Desc 2"},
            ),
        ]

        with patch("sema_tree.builder.WebCrawler") as MockWebCrawler:
            mock_crawler = AsyncMock()
            mock_crawler.crawl = AsyncMock(return_value=mock_pages)
            MockWebCrawler.return_value = mock_crawler

            builder = IndexBuilder(provider=MockProvider())
            index = await builder.build("https://example.com/docs", source_type="website")

        assert len(index.sources) == 1
        assert index.sources[0].type == SourceType.website
        assert index.sources[0].page_count == 2
        assert index.root.id == "0"

    @pytest.mark.asyncio
    async def test_build_raises_on_empty(self):
        with patch("sema_tree.builder.WebCrawler") as MockWebCrawler:
            mock_crawler = AsyncMock()
            mock_crawler.crawl = AsyncMock(return_value=[])
            MockWebCrawler.return_value = mock_crawler

            builder = IndexBuilder(provider=MockProvider())
            with pytest.raises(ValueError, match="No pages found"):
                await builder.build("https://example.com", source_type="website")
