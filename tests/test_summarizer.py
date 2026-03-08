"""Tests for the Summarizer."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from sema_tree.crawlers.base import CrawledPage
from sema_tree.llm import LLMProvider, LLMResponse
from sema_tree.models import RefType
from sema_tree.structurers.base import SkeletonNode
from sema_tree.summarizer import Summarizer


class MockProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, text: str = "Generated summary."):
        self._text = text
        self.call_count = 0

    async def generate(self, user_message, system="", max_tokens=256, temperature=0.0):
        self.call_count += 1
        return LLMResponse(text=self._text)


def _make_leaf(title: str, content: str = "Some content", **metadata) -> SkeletonNode:
    page = CrawledPage(
        url_or_path=f"https://example.com/{title.lower().replace(' ', '-')}",
        title=title,
        content=content,
        ref_type=RefType.url,
        metadata=metadata,
    )
    return SkeletonNode(
        title=title,
        ref=page.url_or_path,
        ref_type=RefType.url,
        page=page,
    )


class TestSummarizer:
    @pytest.mark.asyncio
    async def test_leaf_summarization(self):
        provider = MockProvider("This page covers installation.")
        summarizer = Summarizer(provider=provider)

        skeleton = _make_leaf("Getting Started", "Install the package using pip...")
        result = await summarizer.summarize_tree(skeleton, "test-src", "0.0")

        assert result.summary == "This page covers installation."
        assert result.id == "0.0"
        assert result.title == "Getting Started"
        assert result.ref is not None
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_llms_txt_description_skips_llm(self):
        provider = MockProvider()
        summarizer = Summarizer(provider=provider)

        # Description must be > 50 chars for the summarizer to skip the LLM call
        long_desc = "How to install and set up the tool from scratch using pip or uv on macOS and Linux"
        skeleton = _make_leaf(
            "Getting Started",
            "Full content here...",
            llms_txt_description=long_desc,
        )
        result = await summarizer.summarize_tree(skeleton, "test-src", "0.0")

        # Should use the description directly without calling LLM
        assert result.summary == long_desc
        assert provider.call_count == 0

    @pytest.mark.asyncio
    async def test_branch_summarization(self):
        provider = MockProvider("Section covering setup topics.")
        summarizer = Summarizer(provider=provider)

        skeleton = SkeletonNode(
            title="Setup",
            children=[
                _make_leaf("Install", "Install instructions..."),
                _make_leaf("Configure", "Config instructions..."),
            ],
        )
        result = await summarizer.summarize_tree(skeleton, "test-src", "0.0")

        assert result.summary == "Section covering setup topics."
        assert len(result.children) == 2
        assert result.children[0].id == "0.0.0"
        assert result.children[1].id == "0.0.1"

    @pytest.mark.asyncio
    async def test_hierarchical_ids(self):
        provider = MockProvider("Summary")
        summarizer = Summarizer(provider=provider)

        skeleton = SkeletonNode(
            title="Root",
            children=[
                SkeletonNode(
                    title="A",
                    children=[
                        _make_leaf("A1"),
                        _make_leaf("A2"),
                    ],
                ),
                _make_leaf("B"),
            ],
        )

        result = await summarizer.summarize_tree(skeleton, "src", "0.0")
        assert result.id == "0.0"
        assert result.children[0].id == "0.0.0"
        assert result.children[0].children[0].id == "0.0.0.0"
        assert result.children[0].children[1].id == "0.0.0.1"
        assert result.children[1].id == "0.0.1"

    @pytest.mark.asyncio
    async def test_source_id_propagation(self):
        provider = MockProvider("Summary")
        summarizer = Summarizer(provider=provider)

        skeleton = SkeletonNode(
            title="Root",
            children=[_make_leaf("Page")],
        )
        result = await summarizer.summarize_tree(skeleton, "my-source", "0.0")

        assert result.source_id == "my-source"
        assert result.children[0].source_id == "my-source"

    @pytest.mark.asyncio
    async def test_empty_content_fallback(self):
        provider = MockProvider()
        summarizer = Summarizer(provider=provider)

        page = CrawledPage(
            url_or_path="https://example.com/empty",
            title="Empty Page",
            content="",
            ref_type=RefType.url,
        )
        skeleton = SkeletonNode(
            title="Empty Page",
            ref=page.url_or_path,
            ref_type=RefType.url,
            page=page,
        )
        result = await summarizer.summarize_tree(skeleton, "src", "0.0")
        assert result.summary == "Documentation page: Empty Page"
        assert provider.call_count == 0
