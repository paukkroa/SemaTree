"""Tests for the incremental updater."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, patch

import pytest

from sema_tree.crawlers.base import CrawledPage
from sema_tree.models import SemaTree, IndexNode, RefType, Source, SourceType
from sema_tree.updater import IncrementalUpdater, UpdateDiff


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_page(url: str, content: str, title: str = "Page") -> CrawledPage:
    return CrawledPage(
        url_or_path=url,
        title=title,
        content=content,
        ref_type=RefType.url,
    )


def _sha(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _make_index_with_leaves(
    leaves: list[tuple[str, str, str]]
) -> SemaTree:
    """Build a minimal index with specified (url, content, title) leaves."""
    children = [
        IndexNode(
            id=f"0.0.{i}",
            title=title,
            summary=f"Summary of {title}",
            nav_summary=title,
            ref=url,
            ref_type=RefType.url,
            source_id="src",
            content_hash=_sha(content),
        )
        for i, (url, content, title) in enumerate(leaves)
    ]
    source_root = IndexNode(
        id="0.0",
        title="Source Root",
        summary="All docs",
        nav_summary="All docs",
        source_id="src",
        children=children,
    )
    return SemaTree(
        sources=[Source(id="src", type=SourceType.website, origin="https://example.com", page_count=len(leaves))],
        root=IndexNode(id="0", title="Root", summary="Root", children=[source_root]),
    )


# ---------------------------------------------------------------------------
# UpdateDiff tests
# ---------------------------------------------------------------------------


class TestComputeDiff:
    @pytest.mark.asyncio
    async def test_all_unchanged(self):
        old_index = _make_index_with_leaves([
            ("https://example.com/a", "content A", "A"),
            ("https://example.com/b", "content B", "B"),
        ])
        new_pages = [
            _make_page("https://example.com/a", "content A"),
            _make_page("https://example.com/b", "content B"),
        ]
        updater = IncrementalUpdater()
        diff = await updater.compute_diff(old_index, new_pages)

        assert len(diff.unchanged) == 2
        assert len(diff.changed) == 0
        assert len(diff.added) == 0
        assert len(diff.deleted) == 0
        assert not diff.has_changes

    @pytest.mark.asyncio
    async def test_detects_changed_page(self):
        old_index = _make_index_with_leaves([
            ("https://example.com/a", "old content", "A"),
        ])
        new_pages = [_make_page("https://example.com/a", "new content", "A")]
        updater = IncrementalUpdater()
        diff = await updater.compute_diff(old_index, new_pages)

        assert len(diff.changed) == 1
        assert diff.changed[0].content == "new content"
        assert len(diff.unchanged) == 0
        assert diff.has_changes

    @pytest.mark.asyncio
    async def test_detects_added_page(self):
        old_index = _make_index_with_leaves([
            ("https://example.com/a", "content A", "A"),
        ])
        new_pages = [
            _make_page("https://example.com/a", "content A"),
            _make_page("https://example.com/new", "brand new", "New"),
        ]
        updater = IncrementalUpdater()
        diff = await updater.compute_diff(old_index, new_pages)

        assert len(diff.added) == 1
        assert diff.added[0].url_or_path == "https://example.com/new"

    @pytest.mark.asyncio
    async def test_detects_deleted_page(self):
        old_index = _make_index_with_leaves([
            ("https://example.com/a", "content A", "A"),
            ("https://example.com/gone", "content gone", "Gone"),
        ])
        new_pages = [_make_page("https://example.com/a", "content A")]
        updater = IncrementalUpdater()
        diff = await updater.compute_diff(old_index, new_pages)

        assert "https://example.com/gone" in diff.deleted

    @pytest.mark.asyncio
    async def test_mixed_diff(self):
        old_index = _make_index_with_leaves([
            ("https://example.com/unchanged", "same", "Unchanged"),
            ("https://example.com/changed", "old", "Changed"),
            ("https://example.com/deleted", "bye", "Deleted"),
        ])
        new_pages = [
            _make_page("https://example.com/unchanged", "same"),
            _make_page("https://example.com/changed", "new content"),
            _make_page("https://example.com/added", "hello new"),
        ]
        updater = IncrementalUpdater()
        diff = await updater.compute_diff(old_index, new_pages)

        assert len(diff.unchanged) == 1
        assert len(diff.changed) == 1
        assert len(diff.added) == 1
        assert len(diff.deleted) == 1


# ---------------------------------------------------------------------------
# apply_diff tests (mock LLM summarizer)
# ---------------------------------------------------------------------------


def _mock_summarizer():
    """Return a Summarizer-like mock whose async methods return fake text."""
    mock = AsyncMock()
    mock._summarize_leaf = AsyncMock(return_value=("New detailed summary.", "New nav."))
    mock._summarize_branch = AsyncMock(return_value=("Branch summary.", "Branch nav."))
    return mock


class TestIncrementalUpdateChangedLeaf:
    @pytest.mark.asyncio
    async def test_changed_leaf_summary_updated(self):
        old_index = _make_index_with_leaves([
            ("https://example.com/a", "old content", "A"),
        ])
        new_pages = [_make_page("https://example.com/a", "new content", "A")]
        updater = IncrementalUpdater()
        diff = await updater.compute_diff(old_index, new_pages)

        with patch.object(updater, "_find_leaf_by_ref", wraps=updater._find_leaf_by_ref):
            mock_sum = _mock_summarizer()

            # Patch summarizer creation inside apply_diff
            with patch("sema_tree.updater.Summarizer", return_value=mock_sum):
                result = await updater.apply_diff(old_index, diff, source_id="src")

        leaf = result.root.all_leaves()[0]
        assert leaf.content_hash == _sha("new content")

    @pytest.mark.asyncio
    async def test_unchanged_leaf_not_touched(self):
        old_index = _make_index_with_leaves([
            ("https://example.com/a", "same content", "A"),
            ("https://example.com/b", "other content", "B"),
        ])
        new_pages = [
            _make_page("https://example.com/a", "same content"),
            _make_page("https://example.com/b", "changed content"),
        ]
        updater = IncrementalUpdater()
        diff = await updater.compute_diff(old_index, new_pages)

        original_hash_a = _sha("same content")
        mock_sum = _mock_summarizer()
        with patch("sema_tree.updater.Summarizer", return_value=mock_sum):
            result = await updater.apply_diff(old_index, diff, source_id="src")

        leaves = {leaf.ref: leaf for leaf in result.root.all_leaves()}
        # Unchanged leaf keeps its original hash
        assert leaves["https://example.com/a"].content_hash == original_hash_a
        # Changed leaf gets new hash
        assert leaves["https://example.com/b"].content_hash == _sha("changed content")


class TestIncrementalUpdateAddedLeaf:
    @pytest.mark.asyncio
    async def test_added_leaf_appended(self):
        old_index = _make_index_with_leaves([
            ("https://example.com/a", "content A", "A"),
        ])
        new_pages = [
            _make_page("https://example.com/a", "content A"),
            _make_page("https://example.com/new", "brand new content", "New"),
        ]
        updater = IncrementalUpdater()
        diff = await updater.compute_diff(old_index, new_pages)

        mock_sum = _mock_summarizer()
        with patch("sema_tree.updater.Summarizer", return_value=mock_sum):
            result = await updater.apply_diff(old_index, diff, source_id="src")

        leaves = result.root.all_leaves()
        assert len(leaves) == 2
        refs = {leaf.ref for leaf in leaves}
        assert "https://example.com/new" in refs


class TestIncrementalUpdateDeletedLeaf:
    @pytest.mark.asyncio
    async def test_deleted_leaf_removed(self):
        old_index = _make_index_with_leaves([
            ("https://example.com/a", "content A", "A"),
            ("https://example.com/gone", "going away", "Gone"),
        ])
        new_pages = [_make_page("https://example.com/a", "content A")]
        updater = IncrementalUpdater()
        diff = await updater.compute_diff(old_index, new_pages)

        mock_sum = _mock_summarizer()
        with patch("sema_tree.updater.Summarizer", return_value=mock_sum):
            result = await updater.apply_diff(old_index, diff, source_id="src")

        leaves = result.root.all_leaves()
        assert len(leaves) == 1
        assert leaves[0].ref == "https://example.com/a"


class TestUpdateDiffHasChanges:
    def test_empty_diff_has_no_changes(self):
        diff = UpdateDiff()
        assert not diff.has_changes

    def test_diff_with_changed_has_changes(self):
        diff = UpdateDiff(changed=[_make_page("u", "c")])
        assert diff.has_changes

    def test_diff_with_added_has_changes(self):
        diff = UpdateDiff(added=[_make_page("u", "c")])
        assert diff.has_changes

    def test_diff_with_deleted_has_changes(self):
        diff = UpdateDiff(deleted=["url"])
        assert diff.has_changes

    def test_diff_with_only_unchanged_no_changes(self):
        diff = UpdateDiff(unchanged=["url1", "url2"])
        assert not diff.has_changes
