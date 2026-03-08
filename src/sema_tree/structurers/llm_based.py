"""LLM-based structurer — builds semantic hierarchies via recursive grouping."""

from __future__ import annotations

import json
import logging
from typing import Any

from sema_tree.crawlers.base import CrawledPage
from sema_tree.llm import LLMProvider, get_provider
from sema_tree.models import RefType

from .base import SkeletonNode

logger = logging.getLogger(__name__)

# Pages in a group at or below this threshold become direct leaf children.
_LEAF_THRESHOLD = 7

_SYSTEM_PROMPT = """\
You are an expert information architect. Your job is to organise documentation \
pages into a clear, navigable hierarchy that helps an AI agent quickly find \
relevant information.

Given a numbered list of pages (index, title, optional description), group them \
into {min_groups}-{max_groups} **semantically meaningful categories**. \
Pick category names that describe the *topic* — not the format — of the pages \
inside (e.g. "Getting Started", "IDE Integrations", "Security & Compliance", \
NOT "Pages 1-10" or "Miscellaneous").

Return ONLY valid JSON — no markdown fences, no commentary:
{{
  "categories": [
    {{
      "title": "Category Name",
      "page_indices": [0, 1, 2]
    }}
  ]
}}

Rules:
- Every page index must appear in exactly one category.
- Page indices are 0-based, matching the input list.
- Keep categories balanced — avoid dumping most pages into one category.
- Prefer clear, intuitive labels an LLM agent would understand.
- Do NOT nest subcategories — return a flat list of categories. \
  (Deeper nesting is handled by a second pass.)
- **NEVER** create categories named "Other", "Miscellaneous", "General", \
  "Uncategorized", or any catch-all/dumping-ground category. Every page must \
  be placed into a meaningful, descriptively-named category that reflects its \
  actual topic. If a page does not fit neatly into another group, create a \
  specific category for it (e.g. "Troubleshooting & Debugging" or \
  "IDE Integrations" — not "Other").
- Each category must have at least 2 pages. If you cannot find a second page \
  for a topic, merge it into the closest related category instead.\
"""

_RECURSIVE_SYSTEM_PROMPT = """\
You are an expert information architect. You are sub-grouping pages within \
the "{parent_category}" category into finer sub-categories.

Given a numbered list of pages (index, title, optional description), group them \
into {min_groups}-{max_groups} **semantically meaningful sub-categories** of \
"{parent_category}". \
Pick sub-category names that describe the *topic* — not the format — of the \
pages inside.

Return ONLY valid JSON — no markdown fences, no commentary:
{{
  "categories": [
    {{
      "title": "Sub-Category Name",
      "page_indices": [0, 1, 2]
    }}
  ]
}}

Rules:
- Every page index must appear in exactly one sub-category.
- Page indices are 0-based, matching the input list.
- Keep sub-categories balanced — avoid dumping most pages into one sub-category.
- Prefer clear, intuitive labels an LLM agent would understand.
- Do NOT nest further subcategories — return a flat list. \
  (Deeper nesting is handled by a later pass.)
- **NEVER** create sub-categories named "Other", "Miscellaneous", "General", \
  "Uncategorized", or any catch-all/dumping-ground sub-category. Every page \
  must be placed into a meaningful, descriptively-named sub-category.
- Each sub-category must have at least 2 pages. If you cannot find a second \
  page for a topic, merge it into the closest related sub-category instead.\
"""


def _build_user_message(pages: list[CrawledPage]) -> str:
    lines: list[str] = []
    for i, page in enumerate(pages):
        desc = page.metadata.get("llms_txt_description", "") or page.metadata.get("description", "")
        # Include a short content preview for pages without a description
        if not desc and page.content:
            desc = page.content[:200].replace("\n", " ").strip()
        entry = f"[{i}] {page.title}"
        if desc:
            entry += f" — {desc}"
        lines.append(entry)
    return "Organise these documentation pages into categories:\n\n" + "\n".join(lines)


def _build_flat_tree(pages: list[CrawledPage]) -> SkeletonNode:
    """Fallback: all pages as direct children of root."""
    children = [
        SkeletonNode(
            title=page.title,
            ref=page.url_or_path,
            ref_type=page.ref_type,
            page=page,
        )
        for page in pages
    ]
    return SkeletonNode(title="Root", children=children)


def _extract_json(raw: str) -> dict:
    """Robustly extract the JSON object from potentially noisy LLM output."""
    cleaned = raw.strip()

    # Strip markdown code fences if present
    if "```" in cleaned:
        import re
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

    # Strategy 1: direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: find the first { ... } block containing "categories"
    brace_start = cleaned.find("{")
    if brace_start != -1:
        depth = 0
        for i in range(brace_start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[brace_start:i + 1]
                    try:
                        data = json.loads(candidate)
                        if "categories" in data:
                            return data
                    except json.JSONDecodeError:
                        pass
                    break

    # Strategy 3: regex for anything that looks like a JSON object
    import re
    match = re.search(r'\{[^{}]*"categories"\s*:\s*\[.*\]\s*\}', cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError(f"Could not extract valid JSON from LLM response: {cleaned[:200]}")


def _parse_response(raw: str, pages: list[CrawledPage]) -> SkeletonNode:
    """Parse the LLM JSON response into a single-level SkeletonNode tree.

    Returns a root node whose children are category nodes, each containing
    leaf pages. This is used for one level of the recursive hierarchy.
    """
    data = _extract_json(raw)
    categories: list[dict[str, Any]] = data["categories"]

    root_children: list[SkeletonNode] = []
    seen_indices: set[int] = set()

    for cat in categories:
        cat_children: list[SkeletonNode] = []
        for idx in cat.get("page_indices") or []:
            if not (0 <= idx < len(pages)):
                continue
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            page = pages[idx]
            cat_children.append(
                SkeletonNode(
                    title=page.title,
                    ref=page.url_or_path,
                    ref_type=page.ref_type,
                    page=page,
                )
            )
        if cat_children:
            root_children.append(SkeletonNode(title=cat["title"], children=cat_children))

    # Catch any pages the LLM missed
    missing = [i for i in range(len(pages)) if i not in seen_indices]
    if missing:
        other_children = [
            SkeletonNode(
                title=pages[i].title,
                ref=pages[i].url_or_path,
                ref_type=pages[i].ref_type,
                page=pages[i],
            )
            for i in missing
        ]
        root_children.append(SkeletonNode(title="Other", children=other_children))

    return SkeletonNode(title="Root", children=root_children)


_CATCH_ALL_NAMES = frozenset({
    "other", "miscellaneous", "misc", "general", "uncategorized",
    "everything else", "remaining", "additional",
})


def _is_catch_all(title: str) -> bool:
    """Return True if *title* looks like a catch-all / dumping-ground name."""
    return title.strip().lower() in _CATCH_ALL_NAMES


def _collapse_single_children(root: SkeletonNode) -> SkeletonNode:
    """Collapse category nodes that have only 1 child.

    If a category node (non-leaf, non-root) has exactly one child, that child
    is promoted to replace the parent.  This is applied recursively bottom-up
    so deeply nested single-child chains are fully collapsed.
    """
    if root.is_leaf:
        return root

    # Recurse first so we collapse from the bottom up
    new_children: list[SkeletonNode] = []
    for child in root.children:
        collapsed = _collapse_single_children(child)
        new_children.append(collapsed)
    root.children = new_children

    # Now collapse any category child that itself has exactly 1 child
    final_children: list[SkeletonNode] = []
    for child in root.children:
        if not child.is_leaf and len(child.children) == 1:
            # Promote the single grandchild to this level
            promoted = child.children[0]
            final_children.append(promoted)
        else:
            final_children.append(child)

    root.children = final_children
    return root


def _merge_catch_all_duplicates(root: SkeletonNode) -> SkeletonNode:
    """Merge catch-all subcategories into matching top-level categories.

    If a catch-all category (e.g. "Other") contains subcategories whose names
    match existing top-level categories, those pages are moved into the matching
    top-level category.  Any remaining pages in the catch-all that don't match
    are redistributed: if there are enough to form their own category they stay;
    otherwise they are appended as direct children of the root.
    """
    if root.is_leaf:
        return root

    # Build a map of top-level category names -> node (case-insensitive)
    top_level_map: dict[str, SkeletonNode] = {}
    catch_all_indices: list[int] = []
    for i, child in enumerate(root.children):
        if _is_catch_all(child.title):
            catch_all_indices.append(i)
        else:
            top_level_map[child.title.strip().lower()] = child

    if not catch_all_indices:
        return root

    children_to_keep: list[SkeletonNode] = []
    for i, child in enumerate(root.children):
        if i not in catch_all_indices:
            children_to_keep.append(child)
            continue

        # Process the catch-all category
        remaining_children: list[SkeletonNode] = []
        for sub in child.children:
            target_key = sub.title.strip().lower()
            if target_key in top_level_map:
                # Merge: move leaves from sub into the matching top-level node
                target = top_level_map[target_key]
                if sub.is_leaf:
                    target.children.append(sub)
                else:
                    target.children.extend(sub.children)
            else:
                remaining_children.append(sub)

        # Remaining items that didn't match any top-level category become
        # direct children of root (don't keep the catch-all wrapper)
        for sub in remaining_children:
            if sub.is_leaf:
                children_to_keep.append(sub)
            else:
                # Non-leaf sub-category keeps its own grouping
                children_to_keep.append(sub)

    root.children = children_to_keep
    return root


def _enforce_min_group_size(root: SkeletonNode, min_size: int = 2) -> SkeletonNode:
    """Dissolve sub-categories that have fewer than *min_size* leaf pages.

    Pages from dissolved sub-categories are promoted to the parent level as
    direct leaf children.  Applied recursively bottom-up.
    """
    if root.is_leaf:
        return root

    # Recurse first
    for i, child in enumerate(root.children):
        root.children[i] = _enforce_min_group_size(child, min_size)

    new_children: list[SkeletonNode] = []
    for child in root.children:
        if child.is_leaf:
            new_children.append(child)
        elif child.count_leaves() < min_size:
            # Dissolve: promote all leaves to this level
            new_children.extend(_collect_all_leaves_as_nodes(child))
        else:
            new_children.append(child)

    root.children = new_children
    return root


def _collect_all_leaves_as_nodes(node: SkeletonNode) -> list[SkeletonNode]:
    """Collect all leaf SkeletonNodes from under *node*."""
    if node.is_leaf:
        return [node]
    result: list[SkeletonNode] = []
    for child in node.children:
        result.extend(_collect_all_leaves_as_nodes(child))
    return result


def _postprocess_tree(root: SkeletonNode) -> SkeletonNode:
    """Apply all post-processing steps to a parsed tree.

    Order matters:
    1. Merge catch-all duplicates first (before collapsing removes wrappers).
    2. Enforce minimum group size (dissolve tiny sub-categories).
    3. Collapse single-child wrappers last (clean up leftovers).
    """
    root = _merge_catch_all_duplicates(root)
    root = _enforce_min_group_size(root)
    root = _collapse_single_children(root)
    return root


def _collect_leaves(node: SkeletonNode) -> list[CrawledPage]:
    """Collect all CrawledPage objects from leaf nodes."""
    if node.is_leaf and node.page:
        return [node.page]
    pages: list[CrawledPage] = []
    for child in node.children:
        pages.extend(_collect_leaves(child))
    return pages


class LLMStructurer:
    """Structurer that uses an LLM to recursively group pages into a semantic hierarchy.

    For a set of N pages, this:
    1. Asks the LLM to group them into 3-7 top-level categories.
    2. For any category with more than ``leaf_threshold`` pages,
       recursively sub-groups until clusters are small enough.

    This produces a deep, navigable tree where an agent can drill down
    by topic at each level.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        leaf_threshold: int = _LEAF_THRESHOLD,
        max_depth: int = 4,
    ) -> None:
        self._provider = provider
        self._leaf_threshold = leaf_threshold
        self._max_depth = max_depth

    async def structure(self, pages: list[CrawledPage]) -> SkeletonNode:
        if not pages:
            return SkeletonNode(title="Root")
        if len(pages) <= self._leaf_threshold:
            return _build_flat_tree(pages)

        provider = self._provider or get_provider()
        tree = await self._recursive_structure(provider, pages, depth=0)
        return _postprocess_tree(tree)

    async def _recursive_structure(
        self,
        provider: LLMProvider,
        pages: list[CrawledPage],
        depth: int,
        parent_category: str | None = None,
    ) -> SkeletonNode:
        """Recursively group pages into a semantic hierarchy."""
        # Base case: small enough to be a flat group
        if len(pages) <= self._leaf_threshold or depth >= self._max_depth:
            return _build_flat_tree(pages)

        # Decide group count based on number of pages
        if len(pages) <= 15:
            min_groups, max_groups = 3, 5
        elif len(pages) <= 40:
            min_groups, max_groups = 4, 7
        else:
            min_groups, max_groups = 5, 8

        # Use context-aware prompt for recursive passes
        if parent_category is not None:
            system = _RECURSIVE_SYSTEM_PROMPT.format(
                parent_category=parent_category,
                min_groups=min_groups,
                max_groups=max_groups,
            )
        else:
            system = _SYSTEM_PROMPT.format(min_groups=min_groups, max_groups=max_groups)

        tree = None
        last_error = None
        for attempt in range(2):
            try:
                response = await provider.generate(
                    user_message=_build_user_message(pages),
                    system=system,
                    max_tokens=4096,
                    temperature=0.1 * attempt,  # slightly more creative on retry
                )
                tree = _parse_response(response.text, pages)
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM structuring attempt %d failed at depth %d: %s",
                    attempt + 1, depth, e,
                )

        if tree is None:
            logger.warning(
                "All structuring attempts failed at depth %d, using flat tree",
                depth, exc_info=last_error,
            )
            return _build_flat_tree(pages)

        # Recursively sub-group any large categories
        new_children: list[SkeletonNode] = []
        for child in tree.children:
            child_pages = _collect_leaves(child)
            if len(child_pages) > self._leaf_threshold and depth + 1 < self._max_depth:
                logger.info(
                    "Recursively structuring '%s' (%d pages) at depth %d",
                    child.title, len(child_pages), depth + 1,
                )
                sub_tree = await self._recursive_structure(
                    provider, child_pages, depth + 1,
                    parent_category=child.title,
                )
                # Keep the parent's category title, replace its children
                sub_tree.title = child.title
                new_children.append(sub_tree)
            else:
                new_children.append(child)

        tree.children = new_children
        return tree
