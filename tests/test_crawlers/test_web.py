"""Tests for WebCrawler."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from sema_tree.crawlers.web import WebCrawler
from sema_tree.models import RefType

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _mock_transport(
    responses: dict[str, tuple[int, str, str]],
    redirects: dict[str, str] | None = None,
) -> httpx.AsyncBaseTransport:
    """Create a mock transport that returns canned responses.

    responses: mapping from URL path to (status_code, content_type, body)
    redirects: optional mapping from URL path to redirect-target URL path.
        When a request matches a redirect key, the transport returns a 302
        with a Location header.  The *target* path should exist in
        ``responses`` so the follow-up request resolves.
    """
    redirects = redirects or {}

    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            path = request.url.path
            full_url = str(request.url)

            # Handle redirects first
            for key, target in redirects.items():
                if key == path or key == full_url or full_url.endswith(key):
                    # Build absolute target URL from the request
                    target_url = str(request.url.copy_with(raw_path=target.encode("ascii")))
                    return httpx.Response(
                        status_code=302,
                        headers={
                            "content-type": "text/html",
                            "location": target_url,
                        },
                        content=b"Redirecting...",
                    )

            # Also check full URL for absolute URLs
            for key, (status, ctype, body) in responses.items():
                if key == path or key == full_url or full_url.endswith(key):
                    return httpx.Response(
                        status_code=status,
                        headers={"content-type": ctype},
                        content=body.encode(),
                    )
            return httpx.Response(status_code=404, content=b"Not found")

    return MockTransport()


class TestLlmsTxtParsing:
    @pytest.mark.asyncio
    async def test_parses_llms_txt(self, monkeypatch):
        llms_txt = FIXTURES / "sample_llms_txt.txt"
        content = llms_txt.read_text()

        responses = {
            "/llms.txt": (200, "text/plain", content),
        }
        # Add responses for each linked page (unique content per page for dedup)
        for path in [
            "/docs/en/getting-started",
            "/docs/en/configuration",
            "/docs/en/cli-usage",
            "/docs/en/mcp-servers",
            "/docs/en/hooks",
        ]:
            name = path.rsplit("/", 1)[-1]
            page_html = f"<html><body><h1>Test Page</h1><p>Content for {name}.</p></body></html>"
            responses[path] = (200, "text/html", page_html)

        transport = _mock_transport(responses)

        crawler = WebCrawler(url="https://code.claude.com", max_pages=10)

        # Monkeypatch httpx.AsyncClient to use our mock transport
        original_init = httpx.AsyncClient.__init__

        def patched_init(self, *args, **kwargs):
            kwargs["transport"] = transport
            kwargs.pop("follow_redirects", None)
            original_init(self, *args, follow_redirects=True, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

        pages = await crawler.crawl()
        assert len(pages) == 5
        assert all(p.ref_type == RefType.url for p in pages)

        titles = {p.title for p in pages}
        assert "Getting Started" in titles
        assert "Configuration" in titles

    @pytest.mark.asyncio
    async def test_llms_txt_descriptions(self, monkeypatch):
        llms_txt = (
            "# Docs\n\n"
            "- [Page A](https://example.com/a): Description of page A\n"
            "- [Page B](https://example.com/b): Description of page B\n"
        )
        page_a_html = "<html><body><p>Content for page A</p></body></html>"
        page_b_html = "<html><body><p>Content for page B</p></body></html>"
        responses = {
            "/llms.txt": (200, "text/plain", llms_txt),
            "/a": (200, "text/html", page_a_html),
            "/b": (200, "text/html", page_b_html),
        }
        transport = _mock_transport(responses)

        original_init = httpx.AsyncClient.__init__

        def patched_init(self, *args, **kwargs):
            kwargs["transport"] = transport
            kwargs.pop("follow_redirects", None)
            original_init(self, *args, follow_redirects=True, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

        crawler = WebCrawler(url="https://example.com")
        pages = await crawler.crawl()
        assert len(pages) == 2

        page_a = next(p for p in pages if p.title == "Page A")
        assert page_a.metadata.get("llms_txt_description") == "Description of page A"

    @pytest.mark.asyncio
    async def test_fallback_to_bfs_when_no_llms_txt(self, monkeypatch):
        html = (
            '<html><body><h1>Home</h1>'
            '<a href="/about">About</a>'
            '</body></html>'
        )
        about_html = "<html><body><h1>About</h1><p>About page.</p></body></html>"

        responses = {
            "/llms.txt": (404, "text/plain", ""),
            "/": (200, "text/html", html),
            "/about": (200, "text/html", about_html),
        }
        transport = _mock_transport(responses)

        original_init = httpx.AsyncClient.__init__

        def patched_init(self, *args, **kwargs):
            kwargs["transport"] = transport
            kwargs.pop("follow_redirects", None)
            original_init(self, *args, follow_redirects=True, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

        crawler = WebCrawler(url="https://example.com", max_pages=5)
        pages = await crawler.crawl()
        assert len(pages) >= 1
        assert all(p.ref_type == RefType.url for p in pages)


class TestWebCrawlerHelpers:
    def test_normalize_url(self):
        assert WebCrawler._normalize_url("https://example.com/docs/") == "https://example.com/docs"
        assert WebCrawler._normalize_url("https://example.com/docs#section") == "https://example.com/docs"

    def test_title_from_url(self):
        assert WebCrawler._title_from_url("https://example.com/getting-started") == "Getting Started"
        assert WebCrawler._title_from_url("https://example.com/") == "example.com"

    def test_md_variant(self):
        assert WebCrawler._md_variant("https://example.com/page.html") == "https://example.com/page.md"
        assert WebCrawler._md_variant("https://example.com/page.md") is None
        assert WebCrawler._md_variant("https://example.com/page") is None


class TestSoft404Detection:
    """Tests for _is_soft_404 static method."""

    def test_title_contains_page_not_found(self):
        assert WebCrawler._is_soft_404("Page Not Found", "some content")

    def test_title_contains_404(self):
        assert WebCrawler._is_soft_404("Error 404", "some content")

    def test_content_contains_page_not_found(self):
        assert WebCrawler._is_soft_404(
            "Some Title",
            "Page not found. The page you are looking for does not exist.",
        )

    def test_content_contains_standalone_404(self):
        assert WebCrawler._is_soft_404(
            "Some Title",
            "# 404\n\nThe requested page could not be found.",
        )

    def test_normal_page_not_flagged(self):
        assert not WebCrawler._is_soft_404(
            "Getting Started",
            "Welcome to the documentation. Here is how to get started.",
        )

    def test_404_inside_longer_number_not_flagged(self):
        # "40400" should not trigger the soft-404 check
        assert not WebCrawler._is_soft_404(
            "Error Codes Reference",
            "Error code 40400 means resource not found in the API.",
        )

    @pytest.mark.asyncio
    async def test_bfs_skips_soft_404_page(self, monkeypatch):
        """BFS crawl should skip pages whose content indicates 'Page Not Found'."""
        home_html = (
            '<html><body><h1>Home</h1>'
            '<a href="/docs/real">Real</a>'
            '<a href="/docs/missing">Missing</a>'
            '</body></html>'
        )
        real_html = "<html><body><h1>Real Page</h1><p>Real documentation content.</p></body></html>"
        not_found_html = (
            "<html><body><h1>Page Not Found</h1>"
            "<p>The page you are looking for does not exist.</p></body></html>"
        )

        responses = {
            "/llms.txt": (404, "text/plain", ""),
            "/docs": (200, "text/html", home_html),
            "/docs/real": (200, "text/html", real_html),
            "/docs/missing": (200, "text/html", not_found_html),
        }
        transport = _mock_transport(responses)

        original_init = httpx.AsyncClient.__init__

        def patched_init(self, *args, **kwargs):
            kwargs["transport"] = transport
            kwargs.pop("follow_redirects", None)
            original_init(self, *args, follow_redirects=True, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

        crawler = WebCrawler(url="https://example.com/docs", max_pages=10)
        pages = await crawler.crawl()

        titles = {p.title for p in pages}
        assert "Real Page" in titles
        assert "Page Not Found" not in titles

    @pytest.mark.asyncio
    async def test_llms_txt_skips_soft_404_page(self, monkeypatch):
        """llms.txt path should skip entries whose fetched content is a soft 404."""
        llms_txt = (
            "# Docs\n\n"
            "- [Good Page](https://example.com/good)\n"
            "- [Missing Page](https://example.com/missing)\n"
        )
        good_html = "<html><body><h1>Good Page</h1><p>Useful content here.</p></body></html>"
        not_found_html = (
            "<html><body><h1>Page Not Found</h1>"
            "<p>Sorry, this page does not exist.</p></body></html>"
        )

        responses = {
            "/llms.txt": (200, "text/plain", llms_txt),
            "/good": (200, "text/html", good_html),
            "/missing": (200, "text/html", not_found_html),
        }
        transport = _mock_transport(responses)

        original_init = httpx.AsyncClient.__init__

        def patched_init(self, *args, **kwargs):
            kwargs["transport"] = transport
            kwargs.pop("follow_redirects", None)
            original_init(self, *args, follow_redirects=True, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

        crawler = WebCrawler(url="https://example.com")
        pages = await crawler.crawl()

        assert len(pages) == 1
        assert pages[0].title == "Good Page"


class TestNonDocPageDetection:
    """Tests for _is_non_doc_page static method."""

    def test_github_search_title(self):
        assert WebCrawler._is_non_doc_page(
            "Search code, repositories, users, issues, pull requests...",
            "Some content",
        )

    def test_github_ui_chrome_content(self):
        content = (
            "stars forks repositories pull requests sponsor "
            "Watch this repository for notifications."
        )
        assert WebCrawler._is_non_doc_page("Some Repo", content)

    def test_normal_doc_page_not_flagged(self):
        assert not WebCrawler._is_non_doc_page(
            "Getting Started",
            "Welcome to the documentation. This guide will help you set up your project.",
        )

    def test_page_with_few_github_terms_not_flagged(self):
        # Only 1 GitHub term - not enough to trigger
        assert not WebCrawler._is_non_doc_page(
            "Contributing Guide",
            "Please submit pull requests for any improvements.",
        )

    @pytest.mark.asyncio
    async def test_bfs_skips_github_page(self, monkeypatch):
        """BFS crawl should skip pages that look like GitHub UI chrome."""
        home_html = (
            '<html><body><h1>Docs</h1>'
            '<a href="/docs/real">Real</a>'
            '<a href="/docs/changelog">Changelog</a>'
            '</body></html>'
        )
        real_html = "<html><body><h1>Real Doc</h1><p>Documentation content here.</p></body></html>"
        github_html = (
            "<html><head><title>Search code, repositories, users, issues</title></head>"
            "<body><h1>Search code, repositories, users, issues</h1>"
            "<p>stars forks repositories pull requests sponsor</p></body></html>"
        )

        responses = {
            "/llms.txt": (404, "text/plain", ""),
            "/docs": (200, "text/html", home_html),
            "/docs/real": (200, "text/html", real_html),
            "/docs/changelog": (200, "text/html", github_html),
        }
        transport = _mock_transport(responses)

        original_init = httpx.AsyncClient.__init__

        def patched_init(self, *args, **kwargs):
            kwargs["transport"] = transport
            kwargs.pop("follow_redirects", None)
            original_init(self, *args, follow_redirects=True, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

        crawler = WebCrawler(url="https://example.com/docs", max_pages=10)
        pages = await crawler.crawl()

        titles = {p.title for p in pages}
        assert "Real Doc" in titles
        # The GitHub page should have been filtered out
        assert not any("search code" in t.lower() for t in titles)


class TestFinalUrlDeduplication:
    """Tests for deduplication by final URL after redirects."""

    @pytest.mark.asyncio
    async def test_bfs_deduplicates_by_final_url(self, monkeypatch):
        """Two different request URLs that redirect to the same final URL
        should result in only one page."""
        home_html = (
            '<html><body><h1>Docs Home</h1>'
            '<a href="/docs/overview">Overview</a>'
            '</body></html>'
        )
        overview_html = (
            "<html><body><h1>Overview</h1>"
            "<p>This is the overview page with unique content.</p></body></html>"
        )

        responses = {
            "/llms.txt": (404, "text/plain", ""),
            "/docs/overview": (200, "text/html", overview_html),
        }
        # /docs redirects to /docs/overview
        redirects = {
            "/docs": "/docs/overview",
        }
        transport = _mock_transport(responses, redirects=redirects)

        original_init = httpx.AsyncClient.__init__

        def patched_init(self, *args, **kwargs):
            kwargs["transport"] = transport
            kwargs.pop("follow_redirects", None)
            original_init(self, *args, follow_redirects=True, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

        crawler = WebCrawler(url="https://example.com/docs", max_pages=10)
        pages = await crawler.crawl()

        # We should get exactly one page (the redirect target), not two
        assert len(pages) == 1
        assert pages[0].title == "Overview"

    @pytest.mark.asyncio
    async def test_llms_txt_deduplicates_by_final_url(self, monkeypatch):
        """llms.txt entries that redirect to the same final URL should be deduplicated."""
        llms_txt = (
            "# Docs\n\n"
            "- [Overview](https://example.com/docs)\n"
            "- [Overview Detail](https://example.com/docs/overview)\n"
        )
        overview_html = (
            "<html><body><h1>Overview</h1>"
            "<p>This is the overview page with unique content for dedup test.</p>"
            "</body></html>"
        )

        responses = {
            "/llms.txt": (200, "text/plain", llms_txt),
            "/docs/overview": (200, "text/html", overview_html),
        }
        redirects = {
            "/docs": "/docs/overview",
        }
        transport = _mock_transport(responses, redirects=redirects)

        original_init = httpx.AsyncClient.__init__

        def patched_init(self, *args, **kwargs):
            kwargs["transport"] = transport
            kwargs.pop("follow_redirects", None)
            original_init(self, *args, follow_redirects=True, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

        crawler = WebCrawler(url="https://example.com")
        pages = await crawler.crawl()

        # Both URLs resolve to /docs/overview, so only one page should remain
        assert len(pages) == 1


class TestContentDeduplication:
    """Tests for deduplication by content similarity."""

    def test_content_fingerprint_same_content(self):
        fp1 = WebCrawler._content_fingerprint("Hello world, this is content.")
        fp2 = WebCrawler._content_fingerprint("Hello world, this is content.")
        assert fp1 == fp2

    def test_content_fingerprint_different_content(self):
        fp1 = WebCrawler._content_fingerprint("Hello world, this is content A.")
        fp2 = WebCrawler._content_fingerprint("Hello world, this is content B.")
        assert fp1 != fp2

    def test_content_fingerprint_uses_first_500_chars(self):
        base = "x" * 500
        fp1 = WebCrawler._content_fingerprint(base + "AAAA")
        fp2 = WebCrawler._content_fingerprint(base + "BBBB")
        # Same first 500 chars -> same fingerprint
        assert fp1 == fp2

    @pytest.mark.asyncio
    async def test_bfs_deduplicates_identical_content(self, monkeypatch):
        """BFS should skip pages whose content matches an already-crawled page."""
        identical_body = "<p>This is the exact same documentation content on both pages.</p>"
        home_html = (
            '<html><body><h1>Docs</h1>'
            '<a href="/docs/page-a">Page A</a>'
            '<a href="/docs/page-b">Page B</a>'
            '</body></html>'
        )
        page_a_html = f"<html><body><h1>Page A</h1>{identical_body}</body></html>"
        page_b_html = f"<html><body><h1>Page A</h1>{identical_body}</body></html>"

        responses = {
            "/llms.txt": (404, "text/plain", ""),
            "/docs": (200, "text/html", home_html),
            "/docs/page-a": (200, "text/html", page_a_html),
            "/docs/page-b": (200, "text/html", page_b_html),
        }
        transport = _mock_transport(responses)

        original_init = httpx.AsyncClient.__init__

        def patched_init(self, *args, **kwargs):
            kwargs["transport"] = transport
            kwargs.pop("follow_redirects", None)
            original_init(self, *args, follow_redirects=True, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

        crawler = WebCrawler(url="https://example.com/docs", max_pages=10)
        pages = await crawler.crawl()

        # Home page + one of the duplicate pages = 2 total
        # (the second duplicate should be skipped)
        content_list = [p.content for p in pages]
        # Verify no two pages have the same fingerprint
        fingerprints = [WebCrawler._content_fingerprint(c) for c in content_list]
        assert len(fingerprints) == len(set(fingerprints))

    @pytest.mark.asyncio
    async def test_llms_txt_deduplicates_identical_content(self, monkeypatch):
        """llms.txt path should deduplicate pages with identical content."""
        identical_body = "<p>Exact same content for both entries in the llms.txt list.</p>"
        llms_txt = (
            "# Docs\n\n"
            "- [Page A](https://example.com/a)\n"
            "- [Page B](https://example.com/b)\n"
        )
        responses = {
            "/llms.txt": (200, "text/plain", llms_txt),
            "/a": (200, "text/html", f"<html><body>{identical_body}</body></html>"),
            "/b": (200, "text/html", f"<html><body>{identical_body}</body></html>"),
        }
        transport = _mock_transport(responses)

        original_init = httpx.AsyncClient.__init__

        def patched_init(self, *args, **kwargs):
            kwargs["transport"] = transport
            kwargs.pop("follow_redirects", None)
            original_init(self, *args, follow_redirects=True, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

        crawler = WebCrawler(url="https://example.com")
        pages = await crawler.crawl()

        # Only one should survive deduplication
        assert len(pages) == 1
