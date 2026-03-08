"""Structurers for organizing crawled pages into tree skeletons."""

from __future__ import annotations

from sema_tree.crawlers.base import CrawledPage
from sema_tree.llm import LLMProvider

from .base import SkeletonNode, Structurer
from .llm_based import LLMStructurer
from .path_based import PathBasedStructurer

__all__ = [
    "SkeletonNode",
    "Structurer",
    "LLMStructurer",
    "PathBasedStructurer",
    "auto_select_structurer",
]


def auto_select_structurer(
    pages: list[CrawledPage],
    provider: LLMProvider | None = None,
) -> Structurer:
    """Always use LLM-based semantic structuring.

    The path-based structurer is available for explicit opt-in only (e.g. for
    local folders where the directory layout is already meaningful).
    """
    if not pages:
        return PathBasedStructurer()
    return LLMStructurer(provider=provider)
