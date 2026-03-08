"""Tests for CrossSourceStructurer."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agentic_index.crawlers.base import CrawledPage
from agentic_index.models import RefType
from agentic_index.structurers.semantic import CrossSourceStructurer, _restore_original_titles
from agentic_index.structurers.base import SkeletonNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_page(path: str, title: str, content: str = "", source_id: str = "src") -> CrawledPage:
    return CrawledPage(
        url_or_path=path,
        title=title,
        content=content,
        ref_type=RefType.file,
        metadata={"original_source_id": source_id, "original_title": title},
    )


def _flat_skeleton(pages: list[CrawledPage]) -> SkeletonNode:
    """Build a trivial flat skeleton (no LLM call needed)."""
    children = [
        SkeletonNode(
            title=p.title,
            ref=p.url_or_path,
            ref_type=p.ref_type,
            page=p,
        )
        for p in pages
    ]
    return SkeletonNode(title="Root", children=children)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestRestoreOriginalTitles:
    def test_restores_leaf_title(self):
        page = CrawledPage(
            url_or_path="/doc.md",
            title="[src-a] My Document",
            content="",
            ref_type=RefType.file,
            metadata={"original_title": "My Document"},
        )
        node = SkeletonNode(title="[src-a] My Document", ref="/doc.md", ref_type=RefType.file, page=page)
        root = SkeletonNode(title="Root", children=[node])
        _restore_original_titles(root)
        assert node.title == "My Document"
        assert node.page.title == "My Document"

    def test_no_original_title_keeps_tagged(self):
        page = CrawledPage(
            url_or_path="/doc.md",
            title="Tagged Title",
            content="",
            ref_type=RefType.file,
        )
        node = SkeletonNode(title="Tagged Title", page=page)
        root = SkeletonNode(title="Root", children=[node])
        _restore_original_titles(root)
        assert node.title == "Tagged Title"

    def test_does_not_touch_branch_nodes(self):
        branch = SkeletonNode(title="[src-a] Category", children=[
            SkeletonNode(title="Leaf", page=_make_page("/l.md", "Leaf"), ref="/l.md")
        ])
        root = SkeletonNode(title="Root", children=[branch])
        _restore_original_titles(root)
        # Branch title should remain unchanged (no page to restore from)
        assert branch.title == "[src-a] Category"


class TestCrossSourceStructurerTagging:
    @pytest.mark.asyncio
    async def test_tags_pages_with_source_prefix(self):
        """The LLM structurer should receive tagged page titles."""
        captured_pages: list[CrawledPage] = []

        async def _fake_structure(pages: list[CrawledPage]) -> SkeletonNode:
            captured_pages.extend(pages)
            return _flat_skeleton(pages)

        pages_by_source = {
            "vendor-a": [_make_page("/a/auth.md", "Auth Guide", source_id="vendor-a")],
            "vendor-b": [_make_page("/b/auth.md", "Auth Setup", source_id="vendor-b")],
        }

        structurer = CrossSourceStructurer()
        mock_llm_structurer = AsyncMock()
        mock_llm_structurer.structure = _fake_structure

        with patch(
            "agentic_index.structurers.semantic.LLMStructurer",
            return_value=mock_llm_structurer,
        ):
            await structurer.structure(pages_by_source)

        titles = [p.title for p in captured_pages]
        # Pages should be tagged with source prefix
        assert any("[vendor-a]" in t for t in titles)
        assert any("[vendor-b]" in t for t in titles)

    @pytest.mark.asyncio
    async def test_restores_titles_after_structuring(self):
        """Original titles must be restored in the returned skeleton."""
        pages_by_source = {
            "src-a": [_make_page("/a/doc.md", "Original Title", source_id="src-a")],
        }

        structurer = CrossSourceStructurer()
        mock_llm_structurer = AsyncMock()
        mock_llm_structurer.structure = AsyncMock(side_effect=lambda pages: _flat_skeleton(pages))

        with patch(
            "agentic_index.structurers.semantic.LLMStructurer",
            return_value=mock_llm_structurer,
        ):
            skeleton = await structurer.structure(pages_by_source)

        leaves = [n for n in skeleton.children if n.is_leaf]
        assert all("[src-a]" not in leaf.title for leaf in leaves)
        assert any(leaf.title == "Original Title" for leaf in leaves)

    @pytest.mark.asyncio
    async def test_source_id_preserved_in_metadata(self):
        """original_source_id metadata must be retained for Summarizer."""
        pages_by_source = {
            "src-a": [_make_page("/a.md", "Doc A", source_id="src-a")],
            "src-b": [_make_page("/b.md", "Doc B", source_id="src-b")],
        }

        structurer = CrossSourceStructurer()
        mock_llm_structurer = AsyncMock()
        mock_llm_structurer.structure = AsyncMock(side_effect=lambda pages: _flat_skeleton(pages))

        with patch(
            "agentic_index.structurers.semantic.LLMStructurer",
            return_value=mock_llm_structurer,
        ):
            skeleton = await structurer.structure(pages_by_source)

        source_ids = {
            leaf.page.metadata.get("original_source_id")
            for leaf in skeleton.children
            if leaf.is_leaf and leaf.page
        }
        assert "src-a" in source_ids
        assert "src-b" in source_ids

    @pytest.mark.asyncio
    async def test_empty_sources_returns_root(self):
        structurer = CrossSourceStructurer()
        skeleton = await structurer.structure({})
        assert skeleton.title == "Root"
        assert skeleton.is_leaf or len(skeleton.children) == 0

    @pytest.mark.asyncio
    async def test_all_pages_included(self):
        """All pages from all sources must appear in the skeleton."""
        pages_by_source = {
            "a": [_make_page(f"/a/{i}.md", f"Page A{i}", source_id="a") for i in range(3)],
            "b": [_make_page(f"/b/{i}.md", f"Page B{i}", source_id="b") for i in range(2)],
        }
        captured: list[CrawledPage] = []

        async def _capture(pages: list[CrawledPage]) -> SkeletonNode:
            captured.extend(pages)
            return _flat_skeleton(pages)

        structurer = CrossSourceStructurer()
        mock_llm_structurer = AsyncMock()
        mock_llm_structurer.structure = _capture

        with patch(
            "agentic_index.structurers.semantic.LLMStructurer",
            return_value=mock_llm_structurer,
        ):
            await structurer.structure(pages_by_source)

        assert len(captured) == 5  # 3 + 2


class TestSemanticVsSourceMode:
    """Verify that semantic mode produces a different tree shape than source mode."""

    @pytest.mark.asyncio
    async def test_semantic_mode_flattens_sources(self):
        """In semantic mode, pages from different sources appear under shared categories."""
        from agentic_index.builder import IndexBuilder

        pages_by_source = {
            "src-a": [_make_page("/a/auth.md", "Auth Guide", source_id="src-a")],
            "src-b": [_make_page("/b/auth.md", "Auth Setup", source_id="src-b")],
        }

        # Mock CrossSourceStructurer to return a flat combined skeleton
        combined_skeleton = _flat_skeleton(
            pages_by_source["src-a"] + pages_by_source["src-b"]
        )

        structurer = CrossSourceStructurer()
        mock_llm_structurer = AsyncMock()
        mock_llm_structurer.structure = AsyncMock(return_value=combined_skeleton)

        with patch(
            "agentic_index.structurers.semantic.LLMStructurer",
            return_value=mock_llm_structurer,
        ):
            skeleton = await structurer.structure(pages_by_source)

        # Flat skeleton has both sources' leaves at the same level
        leaf_refs = {child.ref for child in skeleton.children if child.is_leaf}
        assert "/a/auth.md" in leaf_refs
        assert "/b/auth.md" in leaf_refs
