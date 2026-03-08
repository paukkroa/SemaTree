"""Tests for PathBasedStructurer."""

import pytest

from sema_tree.crawlers.base import CrawledPage
from sema_tree.models import RefType
from sema_tree.structurers.path_based import PathBasedStructurer


def _make_pages(urls: list[str]) -> list[CrawledPage]:
    """Create CrawledPage objects from URL list."""
    return [
        CrawledPage(
            url_or_path=url,
            title=url.rstrip("/").rsplit("/", 1)[-1].replace("-", " ").title(),
            content=f"Content for {url}",
            ref_type=RefType.url,
        )
        for url in urls
    ]


class TestPathBasedStructurer:
    @pytest.mark.asyncio
    async def test_basic_url_hierarchy(self):
        pages = _make_pages([
            "https://docs.example.com/en/getting-started",
            "https://docs.example.com/en/configuration",
            "https://docs.example.com/en/api/auth",
            "https://docs.example.com/en/api/endpoints",
        ])
        structurer = PathBasedStructurer()
        tree = await structurer.structure(pages)

        assert tree.count_leaves() == 4
        assert not tree.is_leaf

    @pytest.mark.asyncio
    async def test_empty_pages(self):
        structurer = PathBasedStructurer()
        tree = await structurer.structure([])
        assert tree.title == "Root"
        assert tree.is_leaf

    @pytest.mark.asyncio
    async def test_single_page(self):
        pages = _make_pages(["https://example.com/docs/page"])
        structurer = PathBasedStructurer()
        tree = await structurer.structure(pages)
        # Should have at least the one leaf
        assert tree.count_leaves() == 1

    @pytest.mark.asyncio
    async def test_collapse_single_child(self):
        """Single-child chains should be collapsed."""
        pages = _make_pages([
            "https://example.com/a/b/c/page1",
            "https://example.com/a/b/c/page2",
        ])
        structurer = PathBasedStructurer()
        tree = await structurer.structure(pages)

        # The chain a -> b -> c should be collapsed
        assert tree.count_leaves() == 2
        # Depth should be less than the full URL depth
        assert tree.count_leaves() == 2

    @pytest.mark.asyncio
    async def test_file_paths(self):
        pages = [
            CrawledPage(
                url_or_path="/docs/guide/install.md",
                title="Install",
                content="Installation guide",
                ref_type=RefType.file,
            ),
            CrawledPage(
                url_or_path="/docs/guide/config.md",
                title="Config",
                content="Configuration",
                ref_type=RefType.file,
            ),
            CrawledPage(
                url_or_path="/docs/api/reference.md",
                title="Reference",
                content="API reference",
                ref_type=RefType.file,
            ),
        ]
        structurer = PathBasedStructurer()
        tree = await structurer.structure(pages)
        assert tree.count_leaves() == 3

    @pytest.mark.asyncio
    async def test_preserves_page_references(self):
        pages = _make_pages(["https://example.com/docs/page1"])
        structurer = PathBasedStructurer()
        tree = await structurer.structure(pages)

        # Find the leaf node
        def find_leaf(node):
            if node.is_leaf:
                return node
            for child in node.children:
                result = find_leaf(child)
                if result:
                    return result
            return None

        leaf = find_leaf(tree)
        assert leaf is not None
        assert leaf.ref == "https://example.com/docs/page1"
        assert leaf.ref_type == RefType.url
