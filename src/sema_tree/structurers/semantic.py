"""Cross-source semantic structurer.

Groups pages from *multiple* sources into a single unified hierarchy where
pages from different sources can share categories.  Source identity is
preserved via ``source_id`` on each leaf node.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sema_tree.crawlers.base import CrawledPage
from sema_tree.llm import LLMProvider
from sema_tree.models import RefType
from sema_tree.structurers.llm_based import LLMStructurer

from .base import SkeletonNode

logger = logging.getLogger(__name__)


class CrossSourceStructurer:
    """Groups pages from multiple sources into a single semantic tree.

    Rather than organising the hierarchy by source first
    (``root → source → categories → leaves``), this structurer ignores
    source boundaries and builds one coherent topic tree across all
    sources.  Source identity is preserved in the ``original_source_id``
    metadata field on each leaf's ``CrawledPage`` so that the downstream
    ``Summarizer`` can set ``source_id`` correctly on every ``IndexNode``.

    Usage::

        structurer = CrossSourceStructurer(provider=llm)
        skeleton = await structurer.structure({
            "vendor-a": vendor_a_pages,
            "vendor-b": vendor_b_pages,
        })
    """

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._provider = provider

    async def structure(
        self,
        pages_by_source: dict[str, list[CrawledPage]],
    ) -> SkeletonNode:
        """Build a unified semantic skeleton from pages across multiple sources.

        Args:
            pages_by_source: Mapping of source_id → list of CrawledPage.

        Returns:
            A SkeletonNode tree whose leaves span all sources.
        """
        if not pages_by_source:
            return SkeletonNode(title="Root")

        # Flatten all pages, tagging each with its original source
        all_tagged: list[CrawledPage] = []
        for source_id, pages in pages_by_source.items():
            for page in pages:
                # Tag the title for the LLM prompt so it can disambiguate
                # pages with similar names from different sources.
                tagged = CrawledPage(
                    url_or_path=page.url_or_path,
                    title=f"[{source_id}] {page.title}",
                    content=page.content,
                    ref_type=page.ref_type,
                    metadata={
                        **page.metadata,
                        "original_source_id": source_id,
                        "original_title": page.title,
                    },
                )
                all_tagged.append(tagged)

        logger.info(
            "CrossSourceStructurer: structuring %d pages from %d sources",
            len(all_tagged),
            len(pages_by_source),
        )

        # Use LLMStructurer on the combined (tagged) pool
        structurer = LLMStructurer(provider=self._provider)
        skeleton = await structurer.structure(all_tagged)

        # Restore original (un-tagged) page titles in the skeleton
        _restore_original_titles(skeleton)

        return skeleton


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _restore_original_titles(node: SkeletonNode) -> None:
    """Recursively restore original page titles stripped of the source prefix."""
    if node.is_leaf and node.page is not None:
        original = node.page.metadata.get("original_title")
        if original:
            node.title = original
            # Also restore the page title for correct summaries
            node.page = CrawledPage(
                url_or_path=node.page.url_or_path,
                title=original,
                content=node.page.content,
                ref_type=node.page.ref_type,
                metadata=node.page.metadata,
            )
    for child in node.children:
        _restore_original_titles(child)
