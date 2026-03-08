"""Base structurer protocol and tree skeleton types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from sema_tree.crawlers.base import CrawledPage
from sema_tree.models import RefType


@dataclass
class SkeletonNode:
    """A tree skeleton node (no summaries yet)."""

    title: str
    ref: str | None = None
    ref_type: RefType | None = None
    children: list[SkeletonNode] = field(default_factory=list)
    page: CrawledPage | None = field(default=None, repr=False)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def count_leaves(self) -> int:
        if self.is_leaf:
            return 1
        return sum(c.count_leaves() for c in self.children)


@runtime_checkable
class Structurer(Protocol):
    """Protocol for structuring crawled pages into a tree skeleton."""

    async def structure(self, pages: list[CrawledPage]) -> SkeletonNode:
        """Build a tree skeleton from crawled pages."""
        ...
