"""Multi-source composition for SemaTree."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sema_tree.builder import IndexBuilder, _detect_source_type, _slugify
from sema_tree.models import SemaTree, IndexNode
from sema_tree.summarizer import Summarizer

logger = logging.getLogger(__name__)


async def _regenerate_root_summary(index: SemaTree, summarizer: Summarizer) -> None:
    """Regenerate the root node summary from its children."""
    if not index.root.children:
        index.root.summary = "Empty index with no sources."
        return

    if len(index.root.children) == 1:
        index.root.summary = index.root.children[0].summary
        return

    # Summarize from children
    children_info = "\n".join(
        f"- {child.title}: {child.summary}" for child in index.root.children
    )
    index.root.summary = await summarizer._call_llm(
        "You are a technical documentation summarizer. Given the titles and summaries of indexed sources, write a concise 1-3 sentence overview.",
        f"Indexed sources:\n{children_info}",
    )


async def add_source(
    index: SemaTree,
    source: str,
    builder: IndexBuilder | None = None,
) -> SemaTree:
    """Add a new source to an existing index.

    Builds the source tree and attaches it as a child of root.
    """
    builder = builder or IndexBuilder()
    source_id = _slugify(source)
    source_type = _detect_source_type(source)

    # Determine ID prefix based on existing children count
    child_idx = len(index.root.children)
    id_prefix = f"0.{child_idx}"

    logger.info("Adding source %s (id=%s) at prefix %s", source, source_id, id_prefix)

    source_tree, source_meta, page_count = await builder.build_source_tree(
        source, source_id, source_type, id_prefix
    )

    index.sources.append(source_meta)
    index.root.children.append(source_tree)

    # Regenerate root summary
    summarizer = Summarizer(provider=builder._provider)
    await _regenerate_root_summary(index, summarizer)

    index.updated_at = datetime.now(timezone.utc)
    return index


async def update_source(
    index: SemaTree,
    source_id: str,
    builder: IndexBuilder | None = None,
    incremental: bool = True,
) -> SemaTree:
    """Re-crawl and rebuild (or incrementally update) a source within the index.

    Args:
        index: The current SemaTree.
        source_id: ID of the source to update.
        builder: Optional IndexBuilder (created with default provider if omitted).
        incremental: When True, use change detection to only re-process changed
            pages (default).  When False, perform a full rebuild.
    """
    from sema_tree.crawlers.local import LocalCrawler
    from sema_tree.crawlers.web import WebCrawler
    from sema_tree.models import SourceType

    builder = builder or IndexBuilder()

    # Find existing source
    source_meta = index.find_source(source_id)
    if source_meta is None:
        raise ValueError(f"Source not found: {source_id}")

    # Find the child index for this source
    child_idx = None
    for i, child in enumerate(index.root.children):
        if child.source_id == source_id:
            child_idx = i
            break

    if child_idx is None:
        raise ValueError(f"Source tree not found in index: {source_id}")

    id_prefix = f"0.{child_idx}"
    logger.info("Updating source %s at prefix %s (incremental=%s)", source_id, id_prefix, incremental)

    if incremental:
        # Crawl to get fresh pages
        if source_meta.type == SourceType.website:
            crawler = WebCrawler(url=source_meta.origin)
        else:
            crawler = LocalCrawler(path=source_meta.origin)
        new_pages = await crawler.crawl()
        if not new_pages:
            raise ValueError(f"No pages found at {source_meta.origin}")

        from sema_tree.updater import IncrementalUpdater
        updater = IncrementalUpdater()
        diff = await updater.compute_diff(index, new_pages)

        if not diff.has_changes:
            logger.info("No changes detected for source %s — skipping update", source_id)
            return index

        index = await updater.apply_diff(index, diff, source_id=source_id, builder=builder)

        # Update source metadata
        from datetime import datetime, timezone
        for i, s in enumerate(index.sources):
            if s.id == source_id:
                index.sources[i] = index.sources[i].model_copy(update={
                    "crawled_at": datetime.now(timezone.utc),
                    "page_count": len(new_pages),
                })
                break
    else:
        # Full rebuild
        new_tree, new_meta, page_count = await builder.build_source_tree(
            source_meta.origin, source_id, source_meta.type, id_prefix
        )

        for i, s in enumerate(index.sources):
            if s.id == source_id:
                index.sources[i] = new_meta
                break

        index.root.children[child_idx] = new_tree

    # Regenerate root summary
    summarizer = Summarizer(provider=builder._provider)
    await _regenerate_root_summary(index, summarizer)

    index.updated_at = datetime.now(timezone.utc)
    return index


async def remove_source(
    index: SemaTree,
    source_id: str,
) -> SemaTree:
    """Remove a source and its subtree from the index."""
    # Remove source metadata
    index.sources = [s for s in index.sources if s.id != source_id]

    # Remove source tree from root children
    index.root.children = [
        child for child in index.root.children if child.source_id != source_id
    ]

    # Re-assign child IDs to keep them sequential
    for i, child in enumerate(index.root.children):
        _reassign_ids(child, f"0.{i}")

    # Regenerate root summary
    summarizer = Summarizer()
    await _regenerate_root_summary(index, summarizer)

    index.updated_at = datetime.now(timezone.utc)
    return index


def _reassign_ids(node: IndexNode, new_id: str) -> None:
    """Recursively reassign hierarchical IDs."""
    node.id = new_id
    for i, child in enumerate(node.children):
        _reassign_ids(child, f"{new_id}.{i}")
