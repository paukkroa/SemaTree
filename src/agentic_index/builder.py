"""Index building pipeline orchestrator."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from agentic_index.crawlers.local import LocalCrawler
from agentic_index.crawlers.web import WebCrawler
from agentic_index.llm import LLMProvider, get_provider
from agentic_index.models import AgenticIndex, IndexNode, Source, SourceType
from agentic_index.structurers import auto_select_structurer
from agentic_index.summarizer import Summarizer

logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug for use as source ID."""
    text = text.lower().strip()
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")[:64]


def _detect_source_type(source: str) -> SourceType:
    """Detect whether source is a website URL or local folder path."""
    if source.startswith(("http://", "https://")):
        return SourceType.website
    path = Path(source)
    if path.exists() and path.is_dir():
        return SourceType.local_folder
    if path.exists() and path.is_file():
        return SourceType.local_folder
    # Default to website if it looks like a domain
    if "." in source and "/" in source:
        return SourceType.website
    raise ValueError(f"Cannot detect source type for: {source}")


class IndexBuilder:
    """Orchestrates the full indexing pipeline: Crawl -> Structure -> Summarize -> Assemble."""

    def __init__(self, provider: LLMProvider | None = None):
        self._provider = provider

    async def build(
        self,
        source: str,
        source_type: str = "auto",
        max_pages: int = 200,
        concurrency: int = 10,
    ) -> AgenticIndex:
        """Build a complete index from a single source.

        Args:
            source: URL or filesystem path to index.
            source_type: "website", "local_folder", or "auto" to detect.
            max_pages: Maximum number of pages to crawl (web only).
            concurrency: Number of concurrent requests (web only).

        Returns:
            A fully assembled AgenticIndex.
        """
        # Detect source type
        if source_type == "auto":
            stype = _detect_source_type(source)
        else:
            stype = SourceType(source_type)

        source_id = _slugify(source)
        logger.info("Building index for %s (type=%s, id=%s)", source, stype, source_id)

        # Stage 1: Crawl
        logger.info("Stage 1: Crawling...")
        if stype == SourceType.website:
            crawler = WebCrawler(url=source, max_pages=max_pages, concurrency=concurrency)
        else:
            crawler = LocalCrawler(path=source)

        pages = await crawler.crawl()
        logger.info("Crawled %d pages", len(pages))

        if not pages:
            raise ValueError(f"No pages found at {source}")

        # Stage 2: Structure
        logger.info("Stage 2: Structuring...")
        structurer = auto_select_structurer(pages, provider=self._provider)
        skeleton = await structurer.structure(pages)
        logger.info(
            "Built tree skeleton with %d leaves", skeleton.count_leaves()
        )

        # Stage 3: Summarize
        logger.info("Stage 3: Summarizing...")
        summarizer = Summarizer(provider=self._provider)
        source_tree = await summarizer.summarize_tree(
            skeleton, source_id, id_prefix="0.0"
        )
        logger.info("Summarization complete")

        # Stage 4: Assemble
        logger.info("Stage 4: Assembling index...")
        now = datetime.now(timezone.utc)

        source_meta = Source(
            id=source_id,
            type=stype,
            origin=source,
            crawled_at=now,
            page_count=len(pages),
        )

        root = IndexNode(
            id="0",
            title="Root",
            summary=source_tree.summary,
            children=[source_tree],
        )

        index = AgenticIndex(
            version="1.0",
            created_at=now,
            updated_at=now,
            sources=[source_meta],
            root=root,
        )

        logger.info(
            "Index built: %d sources, %d nodes, %d leaves, depth %d",
            len(index.sources),
            index.root.count_nodes(),
            index.root.count_leaves(),
            index.root.depth,
        )
        return index

    async def build_multi(
        self,
        sources: list[str],
        structure_mode: Literal["source", "semantic"] = "source",
        max_pages: int = 200,
        concurrency: int = 10,
    ) -> AgenticIndex:
        """Build an index from multiple sources.

        Args:
            sources: List of URLs or local paths to index.
            structure_mode: ``"source"`` (default) keeps each source as its own
                top-level subtree.  ``"semantic"`` merges all pages into a single
                cross-source semantic hierarchy.
            max_pages: Maximum pages per source (web only).
            concurrency: Concurrent requests per source (web only).

        Returns:
            A fully assembled AgenticIndex.
        """
        if not sources:
            raise ValueError("At least one source is required")

        now = datetime.now(timezone.utc)

        if structure_mode == "source":
            # Build each source independently and merge at root
            index = AgenticIndex(version="1.0", created_at=now, updated_at=now)
            for source in sources:
                from agentic_index.composer import add_source
                index = await add_source(index, source, builder=self)
            return index

        # --- semantic mode ---
        # 1. Crawl all sources in parallel
        async def _crawl_one(source: str) -> tuple[str, Source, list]:
            stype = _detect_source_type(source)
            source_id = _slugify(source)
            if stype == SourceType.website:
                crawler = WebCrawler(url=source, max_pages=max_pages, concurrency=concurrency)
            else:
                crawler = LocalCrawler(path=source)
            pages = await crawler.crawl()
            if not pages:
                raise ValueError(f"No pages found at {source}")
            meta = Source(
                id=source_id,
                type=stype,
                origin=source,
                crawled_at=datetime.now(timezone.utc),
                page_count=len(pages),
            )
            return source_id, meta, pages

        crawl_tasks = [_crawl_one(s) for s in sources]
        results = await asyncio.gather(*crawl_tasks)

        pages_by_source: dict[str, list] = {}
        all_sources: list[Source] = []
        for source_id, meta, pages in results:
            pages_by_source[source_id] = pages
            all_sources.append(meta)

        # 2. Structure with CrossSourceStructurer
        from agentic_index.structurers.semantic import CrossSourceStructurer
        structurer = CrossSourceStructurer(provider=self._provider)
        skeleton = await structurer.structure(pages_by_source)

        # 3. Summarize (source_id per leaf comes from page.metadata["original_source_id"])
        summarizer = Summarizer(provider=self._provider)
        # Use a placeholder source_id; per-leaf override handles actual assignment
        source_tree = await summarizer.summarize_tree(skeleton, source_id="__multi__", id_prefix="0.0")

        # 4. Assemble
        root = IndexNode(
            id="0",
            title="Root",
            summary=source_tree.summary,
            children=[source_tree],
        )

        return AgenticIndex(
            version="1.0",
            created_at=now,
            updated_at=now,
            sources=all_sources,
            root=root,
        )

    async def build_source_tree(
        self,
        source: str,
        source_id: str,
        source_type: SourceType,
        id_prefix: str = "0.0",
    ) -> tuple[IndexNode, Source, int]:
        """Build just the source tree (used by composer for add/update).

        Returns:
            Tuple of (source_tree, source_metadata, page_count).
        """
        if source_type == SourceType.website:
            crawler = WebCrawler(url=source)
        else:
            crawler = LocalCrawler(path=source)

        pages = await crawler.crawl()
        if not pages:
            raise ValueError(f"No pages found at {source}")

        structurer = auto_select_structurer(pages, provider=self._provider)
        skeleton = await structurer.structure(pages)

        summarizer = Summarizer(provider=self._provider)
        source_tree = await summarizer.summarize_tree(skeleton, source_id, id_prefix)

        source_meta = Source(
            id=source_id,
            type=source_type,
            origin=source,
            crawled_at=datetime.now(timezone.utc),
            page_count=len(pages),
        )

        return source_tree, source_meta, len(pages)
