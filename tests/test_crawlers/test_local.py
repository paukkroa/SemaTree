"""Tests for LocalCrawler."""

from pathlib import Path

import pytest

from sema_tree.crawlers.local import LocalCrawler
from sema_tree.models import RefType


@pytest.fixture
def doc_tree(tmp_path: Path) -> Path:
    """Create a temporary directory tree with sample docs."""
    # Create directory structure
    (tmp_path / "guide").mkdir()
    (tmp_path / "api").mkdir()
    (tmp_path / "node_modules").mkdir()

    # Create doc files
    (tmp_path / "README.md").write_text("# Project\nOverview of the project.")
    (tmp_path / "guide" / "getting-started.md").write_text("# Getting Started\nInstall stuff.")
    (tmp_path / "guide" / "advanced.md").write_text("# Advanced\nAdvanced topics.")
    (tmp_path / "api" / "reference.txt").write_text("API Reference\nEndpoints list.")
    (tmp_path / "api" / "index.html").write_text(
        "<html><body><h1>API</h1><p>API docs.</p></body></html>"
    )

    # Create files that should be ignored
    (tmp_path / "node_modules" / "pkg.md").write_text("Should be ignored")
    (tmp_path / "data.json").write_text('{"not": "a doc"}')

    # Create .gitignore
    (tmp_path / ".gitignore").write_text("*.json\nbuild/\n")

    return tmp_path


class TestLocalCrawler:
    @pytest.mark.asyncio
    async def test_basic_crawl(self, doc_tree: Path):
        crawler = LocalCrawler(path=str(doc_tree))
        pages = await crawler.crawl()
        assert len(pages) == 5  # README, getting-started, advanced, reference, index.html

        # All should be file refs
        assert all(p.ref_type == RefType.file for p in pages)

    @pytest.mark.asyncio
    async def test_ignores_node_modules(self, doc_tree: Path):
        crawler = LocalCrawler(path=str(doc_tree))
        pages = await crawler.crawl()
        filenames = [Path(p.url_or_path).name for p in pages]
        # pkg.md is inside node_modules and should be excluded
        assert "pkg.md" not in filenames

    @pytest.mark.asyncio
    async def test_respects_gitignore(self, doc_tree: Path):
        crawler = LocalCrawler(path=str(doc_tree))
        pages = await crawler.crawl()
        paths = [p.url_or_path for p in pages]
        assert not any(p.endswith(".json") for p in paths)

    @pytest.mark.asyncio
    async def test_extension_filtering(self, doc_tree: Path):
        crawler = LocalCrawler(path=str(doc_tree), extensions=(".md",))
        pages = await crawler.crawl()
        assert all(p.url_or_path.endswith(".md") for p in pages)
        assert len(pages) == 3  # README, getting-started, advanced

    @pytest.mark.asyncio
    async def test_html_to_markdown(self, doc_tree: Path):
        crawler = LocalCrawler(path=str(doc_tree))
        pages = await crawler.crawl()
        html_page = next(p for p in pages if "index.html" in p.url_or_path)
        # Should be converted to markdown
        assert "<html>" not in html_page.content
        assert "API" in html_page.content

    @pytest.mark.asyncio
    async def test_title_from_filename(self, doc_tree: Path):
        crawler = LocalCrawler(path=str(doc_tree))
        pages = await crawler.crawl()
        titles = {p.title for p in pages}
        assert "Readme" in titles or "README" in titles.union({t.upper() for t in titles})
        assert "Getting Started" in titles

    @pytest.mark.asyncio
    async def test_nonexistent_directory(self):
        crawler = LocalCrawler(path="/nonexistent/path")
        with pytest.raises(FileNotFoundError):
            await crawler.crawl()

    @pytest.mark.asyncio
    async def test_metadata_has_relative_path(self, doc_tree: Path):
        crawler = LocalCrawler(path=str(doc_tree))
        pages = await crawler.crawl()
        for page in pages:
            assert "relative_path" in page.metadata
