"""Core data models for SemaTree."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class RefType(str, Enum):
    url = "url"
    file = "file"


class SourceType(str, Enum):
    website = "website"
    local_folder = "local_folder"


class IndexNode(BaseModel):
    """A node in the hierarchical index tree."""

    id: str = Field(description="Dot-separated hierarchical ID, e.g. '0.1.3'")
    title: str = Field(description="Human-readable node name")
    summary: str = Field(default="", description="Detailed AI-generated description")
    nav_summary: str = Field(default="", description="Concise (10-15 word) navigational snippet")
    ref: str | None = Field(default=None, description="URL or file path (null for grouping nodes)")
    ref_type: RefType | None = Field(default=None, description="Type of reference")
    source_id: str | None = Field(default=None, description="ID of the source this belongs to")
    content_hash: str | None = Field(default=None, description="SHA-256 of original page content")
    children: list[IndexNode] = Field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(child.depth for child in self.children)

    def count_leaves(self) -> int:
        if self.is_leaf:
            return 1
        return sum(child.count_leaves() for child in self.children)

    def count_nodes(self) -> int:
        return 1 + sum(child.count_nodes() for child in self.children)

    def find_node(self, node_id: str) -> IndexNode | None:
        if self.id == node_id:
            return self
        for child in self.children:
            result = child.find_node(node_id)
            if result is not None:
                return result
        return None

    def all_leaves(self) -> list[IndexNode]:
        if self.is_leaf:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.all_leaves())
        return leaves


class Source(BaseModel):
    """Metadata for an indexed source."""

    id: str = Field(description="Unique source identifier")
    type: SourceType
    origin: str = Field(description="URL or filesystem path of the source")
    crawled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    page_count: int = Field(default=0, description="Number of pages/files crawled")


class SemaTree(BaseModel):
    """Top-level container for the entire index."""

    version: str = Field(default="1.0")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sources: list[Source] = Field(default_factory=list)
    root: IndexNode = Field(
        default_factory=lambda: IndexNode(id="0", title="Root", summary="")
    )

    def find_source(self, source_id: str) -> Source | None:
        for source in self.sources:
            if source.id == source_id:
                return source
        return None

    def save(self, path: str) -> None:
        from pathlib import Path

        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str) -> SemaTree:
        from pathlib import Path

        return cls.model_validate_json(Path(path).read_text())
