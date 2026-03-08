"""Deterministic path-based structurer using URL/path trie."""

from __future__ import annotations

from urllib.parse import urlparse

from sema_tree.crawlers.base import CrawledPage
from sema_tree.models import RefType

from .base import SkeletonNode


class _TrieNode:
    """Internal trie node for building path hierarchy."""

    __slots__ = ("segment", "children", "pages")

    def __init__(self, segment: str) -> None:
        self.segment = segment
        self.children: dict[str, _TrieNode] = {}
        self.pages: list[CrawledPage] = []

    def insert(self, segments: list[str], page: CrawledPage) -> None:
        node = self
        for seg in segments:
            if seg not in node.children:
                node.children[seg] = _TrieNode(seg)
            node = node.children[seg]
        node.pages.append(page)


def _segment_title(segment: str) -> str:
    """Convert a path segment to a human-readable title."""
    return segment.replace("-", " ").replace("_", " ").strip().title()


def _parse_segments(url_or_path: str) -> list[str]:
    """Extract meaningful path segments from a URL or file path."""
    parsed = urlparse(url_or_path)
    if parsed.scheme in ("http", "https"):
        path = parsed.path
    else:
        path = url_or_path

    segments = [s for s in path.split("/") if s]

    # Strip common file extensions from last segment
    if segments:
        last = segments[-1]
        for ext in (".html", ".htm", ".md", ".txt", ".rst"):
            if last.endswith(ext):
                stripped = last[: -len(ext)]
                if stripped:
                    segments[-1] = stripped
                else:
                    segments.pop()
                break

    return segments


def _trie_to_skeleton(node: _TrieNode, depth: int = 0) -> SkeletonNode:
    """Recursively convert a trie node to a SkeletonNode tree."""
    children: list[SkeletonNode] = []

    # Add child subtrees
    for child in sorted(node.children.values(), key=lambda n: n.segment):
        children.append(_trie_to_skeleton(child, depth + 1))

    # Add leaf nodes for pages stored at this trie node
    for page in node.pages:
        children.append(
            SkeletonNode(
                title=page.title or _segment_title(node.segment),
                ref=page.url_or_path,
                ref_type=page.ref_type,
                page=page,
            )
        )

    title = _segment_title(node.segment) if node.segment else "Root"

    # If this internal node has exactly one page and no child subtrees,
    # promote the page to this node directly
    if len(children) == 1 and children[0].page is not None and not node.children:
        only = children[0]
        return SkeletonNode(
            title=only.title,
            ref=only.ref,
            ref_type=only.ref_type,
            page=only.page,
        )

    return SkeletonNode(title=title, children=children)


def _collapse_single_child(node: SkeletonNode) -> SkeletonNode:
    """Collapse chains of single-child internal nodes to reduce depth."""
    # First recurse into children
    node.children = [_collapse_single_child(c) for c in node.children]

    # Collapse: if this is a non-leaf, non-root grouping node with exactly
    # one child that is also a non-leaf grouping node, merge them.
    while (
        len(node.children) == 1
        and node.ref is None
        and not node.children[0].is_leaf
        and node.children[0].ref is None
    ):
        child = node.children[0]
        node.title = f"{node.title} / {child.title}" if node.title != "Root" else child.title
        node.children = child.children

    return node


class PathBasedStructurer:
    """Deterministic structurer that builds a tree from URL/path hierarchy."""

    async def structure(self, pages: list[CrawledPage]) -> SkeletonNode:
        if not pages:
            return SkeletonNode(title="Root")

        root_trie = _TrieNode("")

        for page in pages:
            segments = _parse_segments(page.url_or_path)
            root_trie.insert(segments, page)

        skeleton = _trie_to_skeleton(root_trie)
        skeleton = _collapse_single_child(skeleton)
        return skeleton
