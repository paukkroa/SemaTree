"""Incremental update engine for AgenticIndex.

Detects which pages changed since the last crawl and updates only the
affected nodes, avoiding a full re-crawl and re-summarize cycle.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

from agentic_index.crawlers.base import CrawledPage
from agentic_index.models import AgenticIndex, IndexNode
from agentic_index.summarizer import Summarizer

logger = logging.getLogger(__name__)


@dataclass
class UpdateDiff:
    """Result of comparing an old index against a fresh crawl."""

    unchanged: list[str] = field(default_factory=list)
    """refs (url/path) of pages whose content hash is identical."""

    changed: list[CrawledPage] = field(default_factory=list)
    """Pages that exist in old index but have different content."""

    added: list[CrawledPage] = field(default_factory=list)
    """Pages present in new crawl but not in the old index."""

    deleted: list[str] = field(default_factory=list)
    """refs present in old index but absent from the new crawl."""

    @property
    def has_changes(self) -> bool:
        return bool(self.changed or self.added or self.deleted)


class IncrementalUpdater:
    """Compute diffs and apply incremental updates to an AgenticIndex."""

    # ------------------------------------------------------------------
    # Diff computation
    # ------------------------------------------------------------------

    async def compute_diff(
        self,
        old_index: AgenticIndex,
        new_pages: list[CrawledPage],
    ) -> UpdateDiff:
        """Compare ``old_index`` leaves against ``new_pages`` by content hash.

        Pages are matched by their ``url_or_path`` (the ``ref`` field on
        IndexNode).  A page is considered *unchanged* when its SHA-256 hash
        matches the stored ``content_hash``; *changed* when the ref exists but
        the hash differs; *added* when the ref is brand new; *deleted* when it
        was in the old index but is absent from the new crawl.
        """
        # Build ref → leaf map from old index
        old_leaves: dict[str, IndexNode] = {}
        for leaf in old_index.root.all_leaves():
            if leaf.ref:
                old_leaves[leaf.ref] = leaf

        new_refs: set[str] = set()
        unchanged: list[str] = []
        changed: list[CrawledPage] = []
        added: list[CrawledPage] = []

        for page in new_pages:
            ref = page.url_or_path
            new_refs.add(ref)
            new_hash = hashlib.sha256(page.content.encode()).hexdigest()

            if ref in old_leaves:
                old_leaf = old_leaves[ref]
                if old_leaf.content_hash and old_leaf.content_hash == new_hash:
                    unchanged.append(ref)
                else:
                    changed.append(page)
            else:
                added.append(page)

        deleted = [ref for ref in old_leaves if ref not in new_refs]

        diff = UpdateDiff(
            unchanged=unchanged,
            changed=changed,
            added=added,
            deleted=deleted,
        )

        logger.info(
            "Diff result: %d unchanged, %d changed, %d added, %d deleted",
            len(unchanged), len(changed), len(added), len(deleted),
        )
        return diff

    # ------------------------------------------------------------------
    # Apply diff
    # ------------------------------------------------------------------

    async def apply_diff(
        self,
        index: AgenticIndex,
        diff: UpdateDiff,
        source_id: str,
        builder: object | None = None,
    ) -> AgenticIndex:
        """Apply an *UpdateDiff* to *index* in-place and return it.

        Strategy:
        * **Unchanged**: no action.
        * **Deleted**: remove leaf node from tree.
        * **Changed**: replace leaf content + update content_hash.
        * **Added**: append leaf to source root (best-effort placement;
          re-structure with ``--restructure`` for proper placement).

        Branch summaries for any modified subtree are regenerated after
        all leaf changes are applied.
        """
        if builder is not None:
            summarizer = Summarizer(provider=getattr(builder, "_provider", None))
        else:
            summarizer = Summarizer()

        dirty_parents: set[str] = set()

        # --- Deletions ---
        for ref in diff.deleted:
            parent_id = self._remove_leaf_by_ref(index.root, ref)
            if parent_id:
                dirty_parents.add(parent_id)
                logger.debug("Deleted leaf with ref=%s", ref)

        # --- Changes: re-summarize leaf ---
        for page in diff.changed:
            node, parent_id = self._find_leaf_by_ref(index.root, page.url_or_path)
            if node is None:
                continue
            new_summary, new_nav = await summarizer._summarize_leaf(
                # Wrap in a SkeletonNode-like object
                _PageWrapper(page)
            )
            new_hash = hashlib.sha256(page.content.encode()).hexdigest()
            node.summary = new_summary
            node.nav_summary = new_nav
            node.content_hash = new_hash
            if parent_id:
                dirty_parents.add(parent_id)
            logger.debug("Updated changed leaf ref=%s", page.url_or_path)

        # --- Additions: place under source root ---
        if diff.added:
            source_root = self._find_source_root(index, source_id)
            for page in diff.added:
                from agentic_index.structurers.base import SkeletonNode
                skeleton_leaf = SkeletonNode(
                    title=page.title,
                    ref=page.url_or_path,
                    ref_type=page.ref_type,
                    page=page,
                )
                summary, nav = await summarizer._summarize_leaf(skeleton_leaf)
                new_hash = hashlib.sha256(page.content.encode()).hexdigest()

                # Build a unique ID for the new leaf
                if source_root is not None:
                    new_id = f"{source_root.id}.{len(source_root.children)}"
                    new_node = IndexNode(
                        id=new_id,
                        title=page.title,
                        summary=summary,
                        nav_summary=nav,
                        ref=page.url_or_path,
                        ref_type=page.ref_type,
                        source_id=source_id,
                        content_hash=new_hash,
                    )
                    source_root.children.append(new_node)
                    dirty_parents.add(source_root.id)
                    logger.debug("Added new leaf ref=%s under source root %s", page.url_or_path, source_root.id)

        # --- Re-summarize dirty branch nodes (bottom-up) ---
        if dirty_parents:
            await self._resync_dirty_branches(index.root, dirty_parents, summarizer)

        return index

    # ------------------------------------------------------------------
    # Tree helpers
    # ------------------------------------------------------------------

    def _find_leaf_by_ref(
        self, root: IndexNode, ref: str
    ) -> tuple[IndexNode | None, str | None]:
        """Return (leaf, parent_id) for the node whose ref matches."""
        return self._find_leaf_recursive(root, ref, parent_id=None)

    def _find_leaf_recursive(
        self, node: IndexNode, ref: str, parent_id: str | None
    ) -> tuple[IndexNode | None, str | None]:
        if node.is_leaf:
            if node.ref == ref:
                return node, parent_id
            return None, None
        for child in node.children:
            result, pid = self._find_leaf_recursive(child, ref, node.id)
            if result is not None:
                return result, pid
        return None, None

    def _remove_leaf_by_ref(self, root: IndexNode, ref: str) -> str | None:
        """Remove a leaf node with the given ref. Returns the parent's ID."""
        return self._remove_recursive(root, ref)

    def _remove_recursive(self, node: IndexNode, ref: str) -> str | None:
        new_children = []
        removed = False
        for child in node.children:
            if child.is_leaf and child.ref == ref:
                removed = True
            else:
                new_children.append(child)
        if removed:
            node.children = new_children
            return node.id
        for child in node.children:
            result = self._remove_recursive(child, ref)
            if result is not None:
                return result
        return None

    def _find_source_root(
        self, index: AgenticIndex, source_id: str
    ) -> IndexNode | None:
        """Find the top-level source node matching source_id."""
        for child in index.root.children:
            if child.source_id == source_id:
                return child
        return None

    async def _resync_dirty_branches(
        self,
        root: IndexNode,
        dirty_ids: set[str],
        summarizer: object,
    ) -> None:
        """Re-summarize all branch nodes whose IDs are in dirty_ids (bottom-up)."""
        await self._resync_recursive(root, dirty_ids, summarizer)

    async def _resync_recursive(
        self,
        node: IndexNode,
        dirty_ids: set[str],
        summarizer: object,
    ) -> bool:
        """Recurse depth-first; re-summarize a node if it's dirty."""
        if node.is_leaf:
            return False

        any_child_dirty = False
        for child in node.children:
            if await self._resync_recursive(child, dirty_ids, summarizer):
                any_child_dirty = True

        if node.id in dirty_ids or any_child_dirty:
            if node.children:
                new_summary, new_nav = await summarizer._summarize_branch(
                    node.title, node.children
                )
                node.summary = new_summary
                node.nav_summary = new_nav
            return True

        return False


class _PageWrapper:
    """Minimal SkeletonNode-like wrapper around a CrawledPage for summarization."""

    def __init__(self, page: CrawledPage) -> None:
        self.page = page
        self.title = page.title
        self.ref = page.url_or_path
        self.ref_type = page.ref_type
        self.children: list = []

    @property
    def is_leaf(self) -> bool:
        return True
