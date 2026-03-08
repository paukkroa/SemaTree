"""Download and cache documentation pages as local markdown files."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify

from evaluation.config import CORPUS_CACHE_DIR, CORPUS_SITEMAP_URLS

logger = logging.getLogger(__name__)


async def fetch_page(client: httpx.AsyncClient, url: str) -> tuple[str, str]:
    """Fetch a single page and return ``(url, html)``."""
    resp = await client.get(url, follow_redirects=True)
    resp.raise_for_status()
    return url, resp.text


def _html_to_markdown(html: str) -> str:
    """Convert raw HTML to clean markdown, stripping nav/footer boilerplate."""
    soup = BeautifulSoup(html, "html.parser")
    # Remove script, style, nav, footer, header elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    # Try to find the main content area
    main = soup.find("main") or soup.find("article") or soup.find("div", class_="content")
    target = main if main else soup.body if soup.body else soup
    return markdownify(str(target), heading_style="ATX", strip=["img"]).strip()


def _slug_from_url(url: str) -> str:
    """Derive a filesystem-safe slug from a URL."""
    from urllib.parse import urlparse

    path = urlparse(url).path.strip("/")
    return path.replace("/", "_") or "index"


async def fetch_corpus(
    urls: list[str] | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
) -> list[Path]:
    """Download doc pages and cache as markdown files.

    Returns list of paths to cached markdown files.
    """
    urls = urls or CORPUS_SITEMAP_URLS
    cache_dir = cache_dir or CORPUS_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_paths: list[Path] = []
    urls_to_fetch: list[str] = []

    for url in urls:
        slug = _slug_from_url(url)
        path = cache_dir / f"{slug}.md"
        if path.exists() and not force:
            logger.info("Using cached %s", path.name)
            cached_paths.append(path)
        else:
            urls_to_fetch.append(url)

    if urls_to_fetch:
        logger.info("Fetching %d pages...", len(urls_to_fetch))
        headers = {
            "User-Agent": "SemaTree/0.1"
        }
        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            tasks = [fetch_page(client, url) for url in urls_to_fetch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error("Failed to fetch page: %s", result)
                continue
            url, html = result
            slug = _slug_from_url(url)
            md = _html_to_markdown(html)
            path = cache_dir / f"{slug}.md"
            path.write_text(md, encoding="utf-8")
            logger.info("Cached %s (%d chars)", path.name, len(md))
            cached_paths.append(path)

    return sorted(cached_paths)
