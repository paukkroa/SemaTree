"""Web crawler with llms.txt fast-path and BFS fallback."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify

from sema_tree.crawlers.base import CrawledPage
from sema_tree.models import RefType

logger = logging.getLogger(__name__)

_LLMS_TXT_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


class WebCrawler:
    """Crawl a website, preferring llms.txt when available.

    Fast path: fetch ``{base_url}/llms.txt`` and parse markdown links.
    Fallback:  BFS crawl starting from *base_url*, staying on the same domain.
    """

    def __init__(
        self,
        url: str,
        max_pages: int = 200,
        concurrency: int = 10,
    ) -> None:
        self.base_url = url.rstrip("/")
        self.max_pages = max_pages
        self.concurrency = concurrency
        parsed = urlparse(self.base_url)
        self._domain = parsed.netloc
        # Restrict BFS crawl to URLs under this path prefix
        self._path_prefix = parsed.path.rstrip("/") or ""

    async def crawl(self) -> list[CrawledPage]:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(30.0),
            headers={"User-Agent": "SemaTree/0.1"},
        ) as client:
            pages = await self._try_llms_txt(client)
            if pages:
                logger.info("llms.txt yielded %d pages", len(pages))
                return pages[: self.max_pages]

            logger.info("No llms.txt found, falling back to BFS crawl")
            return await self._bfs_crawl(client)

    # ------------------------------------------------------------------
    # llms.txt fast path
    # ------------------------------------------------------------------

    async def _try_llms_txt(self, client: httpx.AsyncClient) -> list[CrawledPage]:
        llms_url = f"{self.base_url}/llms.txt"
        try:
            resp = await client.get(llms_url)
            if resp.status_code != 200:
                return []
        except httpx.HTTPError:
            return []

        text = resp.text
        if not _LLMS_TXT_LINK_RE.search(text):
            return []

        links = _LLMS_TXT_LINK_RE.findall(text)
        logger.info("Parsed %d links from llms.txt", len(links))

        # Build (title, absolute_url, description) tuples.
        entries: list[tuple[str, str, str]] = []
        for title, href in links:
            abs_url = urljoin(llms_url, href)
            entries.append((title.strip(), abs_url, ""))

        # Attempt to extract per-link descriptions from surrounding text.
        entries = self._enrich_descriptions(text, entries)

        sem = asyncio.Semaphore(self.concurrency)
        tasks = [
            self._fetch_llms_entry(client, sem, title, url, desc)
            for title, url, desc in entries[: self.max_pages]
        ]
        results = await asyncio.gather(*tasks)

        # Post-filter: deduplicate by final URL and content fingerprint
        seen_final_urls: set[str] = set()
        seen_fingerprints: set[str] = set()
        filtered: list[CrawledPage] = []
        for page in results:
            if page is None:
                continue
            # Deduplicate by final URL stored in metadata
            final_url = page.metadata.pop("_final_url", page.url_or_path)
            norm_final = self._normalize_url(final_url)
            if norm_final in seen_final_urls:
                logger.debug(
                    "Skipping duplicate llms.txt entry (same final URL): %s",
                    page.url_or_path,
                )
                continue
            seen_final_urls.add(norm_final)
            # Deduplicate by content fingerprint
            fp = self._content_fingerprint(page.content)
            if fp in seen_fingerprints:
                logger.debug(
                    "Skipping duplicate llms.txt entry (same content): %s",
                    page.url_or_path,
                )
                continue
            seen_fingerprints.add(fp)
            filtered.append(page)

        return filtered

    @staticmethod
    def _enrich_descriptions(
        text: str,
        entries: list[tuple[str, str, str]],
    ) -> list[tuple[str, str, str]]:
        """Try to extract the text following each link as a description."""
        enriched: list[tuple[str, str, str]] = []
        for title, url, _ in entries:
            # Look for `: description` after the link on the same line.
            escaped_title = re.escape(title)
            escaped_url = re.escape(url)
            pattern = rf"\[{escaped_title}\]\({escaped_url}\):?\s*(.+)"
            match = re.search(pattern, text)
            desc = match.group(1).strip() if match else ""
            enriched.append((title, url, desc))
        return enriched

    async def _fetch_llms_entry(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        title: str,
        url: str,
        description: str,
    ) -> CrawledPage | None:
        async with sem:
            try:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return None
            except httpx.HTTPError:
                logger.debug("Failed to fetch llms.txt entry: %s", url)
                return None

            content_type = resp.headers.get("content-type", "")
            if "text/html" in content_type:
                content = self._html_to_markdown(resp.text)
            else:
                content = resp.text

            # Skip soft-404 pages
            if self._is_soft_404(title, content):
                logger.debug("Skipping soft-404 llms.txt entry: %s", url)
                return None

            # Skip non-documentation pages
            if self._is_non_doc_page(title, content):
                logger.debug("Skipping non-doc llms.txt entry: %s", url)
                return None

            metadata: dict[str, str] = {}
            if description:
                metadata["llms_txt_description"] = description
            # Store final URL for deduplication (will be removed in post-filter)
            metadata["_final_url"] = str(resp.url)

            return CrawledPage(
                url_or_path=url,
                title=title,
                content=content,
                ref_type=RefType.url,
                metadata=metadata,
            )

    # ------------------------------------------------------------------
    # BFS crawl fallback
    # ------------------------------------------------------------------

    async def _bfs_crawl(self, client: httpx.AsyncClient) -> list[CrawledPage]:
        visited: set[str] = set()
        visited_final_urls: set[str] = set()
        seen_fingerprints: set[str] = set()
        queue: asyncio.Queue[str] = asyncio.Queue()
        queue.put_nowait(self.base_url)
        pages: list[CrawledPage] = []
        sem = asyncio.Semaphore(self.concurrency)

        async def worker() -> None:
            while True:
                try:
                    url = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                normalized = self._normalize_url(url)
                if normalized in visited:
                    queue.task_done()
                    continue
                if len(visited) >= self.max_pages:
                    queue.task_done()
                    return

                visited.add(normalized)

                async with sem:
                    page, links, final_url = await self._fetch_page(client, normalized)

                if page is not None:
                    # Deduplicate by final URL (after redirects)
                    norm_final = self._normalize_url(final_url) if final_url else normalized
                    if norm_final in visited_final_urls and norm_final != normalized:
                        logger.debug(
                            "Skipping duplicate (same final URL): %s -> %s",
                            normalized,
                            norm_final,
                        )
                        page = None
                    else:
                        visited_final_urls.add(norm_final)

                if page is not None:
                    # Deduplicate by content similarity
                    fp = self._content_fingerprint(page.content)
                    if fp in seen_fingerprints:
                        logger.debug(
                            "Skipping duplicate (same content fingerprint): %s",
                            normalized,
                        )
                        page = None
                    else:
                        seen_fingerprints.add(fp)

                if page is not None:
                    pages.append(page)

                for link in links:
                    norm_link = self._normalize_url(link)
                    if norm_link not in visited and len(visited) + queue.qsize() < self.max_pages * 2:
                        queue.put_nowait(link)

                queue.task_done()

        # Run BFS in waves until the queue is empty or we hit max_pages.
        while not queue.empty() and len(pages) < self.max_pages:
            batch_size = min(queue.qsize(), self.concurrency)
            workers = [asyncio.create_task(worker()) for _ in range(batch_size)]
            await asyncio.gather(*workers)

        return pages[: self.max_pages]

    async def _fetch_page(
        self,
        client: httpx.AsyncClient,
        url: str,
    ) -> tuple[CrawledPage | None, list[str], str | None]:
        """Fetch a single page; return (page, discovered_links, final_url)."""
        # Try fetching an .md version first (common on doc sites).
        md_url = self._md_variant(url)
        if md_url:
            try:
                resp = await client.get(md_url)
                if resp.status_code == 200 and "text/html" not in resp.headers.get(
                    "content-type", ""
                ):
                    title = self._title_from_url(url)
                    content = resp.text
                    if self._is_soft_404(title, content):
                        logger.debug("Skipping soft-404 page: %s", url)
                        return None, [], str(resp.url)
                    page = CrawledPage(
                        url_or_path=url,
                        title=title,
                        content=content,
                        ref_type=RefType.url,
                    )
                    # Still fetch HTML to extract links.
                    links = await self._extract_links(client, url)
                    return page, links, str(resp.url)
            except httpx.HTTPError:
                pass

        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None, [], None
        except httpx.HTTPError:
            logger.debug("Failed to fetch: %s", url)
            return None, [], None

        # If the response redirected outside our path prefix, skip it
        final_url = str(resp.url)
        final_parsed = urlparse(final_url)
        if final_parsed.netloc != self._domain:
            logger.debug("Skipping redirect outside domain: %s -> %s", url, final_url)
            return None, [], final_url
        if self._path_prefix and not final_parsed.path.startswith(self._path_prefix):
            logger.debug("Skipping redirect outside path prefix: %s -> %s", url, final_url)
            return None, [], final_url

        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type:
            return None, [], final_url

        html = resp.text
        soup = BeautifulSoup(html, "html.parser")
        title = self._extract_title(soup, final_url)
        content = self._html_to_markdown(html)

        # Skip soft-404 pages
        if self._is_soft_404(title, content):
            logger.debug("Skipping soft-404 page: %s", final_url)
            links = self._extract_same_domain_links(soup, final_url)
            return None, links, final_url

        # Skip non-documentation pages (e.g. GitHub UI chrome)
        if self._is_non_doc_page(title, content):
            logger.debug("Skipping non-doc page: %s", final_url)
            return None, [], final_url

        links = self._extract_same_domain_links(soup, final_url)

        page = CrawledPage(
            url_or_path=final_url,
            title=title,
            content=content,
            ref_type=RefType.url,
        )
        return page, links, final_url

    async def _extract_links(self, client: httpx.AsyncClient, url: str) -> list[str]:
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return []
        except httpx.HTTPError:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        return self._extract_same_domain_links(soup, url)

    def _extract_same_domain_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        links: list[str] = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            abs_url = urljoin(base_url, href)
            parsed = urlparse(abs_url)
            # Stay on same domain, strip fragments, skip non-http.
            if parsed.netloc != self._domain:
                continue
            if parsed.scheme not in ("http", "https"):
                continue
            # Stay within the starting URL path prefix
            if self._path_prefix and not parsed.path.startswith(self._path_prefix):
                continue
            # Skip infrastructure paths
            if "/cdn-cgi/" in parsed.path:
                continue
            clean = parsed._replace(fragment="").geturl()
            links.append(clean)
        return links

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _html_to_markdown(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        # Remove script and style tags before conversion.
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        # Search for common documentation content containers
        main = (
            soup.find("main") or 
            soup.find("article") or 
            soup.find("div", class_=re.compile(r"(content|documentation|markdown-body|doc-content|post-content)")) or
            soup.find("div", id=re.compile(r"(content|documentation|main)")) or
            soup.find("body") or 
            soup
        )
        return markdownify(str(main), heading_style="ATX", strip=["img"]).strip()

    @staticmethod
    def _extract_title(soup: BeautifulSoup, url: str) -> str:
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)
        title_tag = soup.find("title")
        if title_tag and title_tag.get_text(strip=True):
            return title_tag.get_text(strip=True)
        return WebCrawler._title_from_url(url)

    @staticmethod
    def _title_from_url(url: str) -> str:
        path = urlparse(url).path.rstrip("/")
        if not path or path == "/":
            return urlparse(url).netloc
        segment = path.rsplit("/", 1)[-1]
        # Remove file extension and convert hyphens/underscores to spaces.
        name = segment.rsplit(".", 1)[0]
        return name.replace("-", " ").replace("_", " ").title()

    @staticmethod
    def _normalize_url(url: str) -> str:
        parsed = urlparse(url)
        # Strip trailing slash and fragment.
        path = parsed.path.rstrip("/") or "/"
        return parsed._replace(path=path, fragment="").geturl()

    @staticmethod
    def _md_variant(url: str) -> str | None:
        """Return an .md variant URL if the original looks like an HTML doc page."""
        parsed = urlparse(url)
        path = parsed.path
        if path.endswith(".md"):
            return None
        if path.endswith(".html") or path.endswith(".htm"):
            md_path = path.rsplit(".", 1)[0] + ".md"
            return parsed._replace(path=md_path).geturl()
        return None

    # ------------------------------------------------------------------
    # Page quality filters
    # ------------------------------------------------------------------

    @staticmethod
    def _is_soft_404(title: str, content: str) -> bool:
        """Detect pages that returned HTTP 200 but are really 'not found' pages."""
        title_lower = title.lower()
        if "page not found" in title_lower or "404" in title_lower:
            return True
        # Check the first portion of content for prominent not-found signals.
        head = content[:1000].lower()
        if "page not found" in head:
            return True
        # Match a standalone "404" (not inside a longer number like a port).
        if re.search(r"\b404\b", head):
            return True
        return False

    @staticmethod
    def _is_non_doc_page(title: str, content: str) -> bool:
        """Detect pages that are GitHub UI / repository chrome rather than docs."""
        title_lower = title.lower()
        non_doc_signals = [
            "search code, repositories",
            "search code, repositories, users",
        ]
        for signal in non_doc_signals:
            if signal in title_lower:
                return True

        # Count GitHub-specific UI terms in the first portion of content.
        head = content[:2000].lower()
        github_terms = ["stars", "forks", "repositories", "pull requests", "sponsor"]
        hits = sum(1 for term in github_terms if term in head)
        if hits >= 3:
            return True

        return False

    @staticmethod
    def _content_fingerprint(content: str) -> str:
        """Return a short hash of the first 500 characters for deduplication."""
        prefix = content[:500]
        return hashlib.sha256(prefix.encode()).hexdigest()
