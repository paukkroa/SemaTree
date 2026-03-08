"""Base crawler protocol and data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from sema_tree.models import RefType


@dataclass
class CrawledPage:
    """A single crawled page/document."""

    url_or_path: str
    title: str
    content: str
    ref_type: RefType
    metadata: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class Crawler(Protocol):
    """Protocol for all crawlers."""

    async def crawl(self) -> list[CrawledPage]:
        """Crawl the source and return a list of pages."""
        ...
