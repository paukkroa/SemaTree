"""Tests for LLMStructurer."""

from __future__ import annotations

import json

import pytest

from sema_tree.crawlers.base import CrawledPage
from sema_tree.llm import LLMProvider, LLMResponse
from sema_tree.models import RefType
from sema_tree.structurers.base import SkeletonNode
from sema_tree.structurers.llm_based import (
    LLMStructurer,
    _collapse_single_children,
    _enforce_min_group_size,
    _is_catch_all,
    _merge_catch_all_duplicates,
    _parse_response,
    _postprocess_tree,
)


def _make_pages(n: int) -> list[CrawledPage]:
    return [
        CrawledPage(
            url_or_path=f"https://example.com/page{i}",
            title=f"Page {i}",
            content=f"Content for page {i}",
            ref_type=RefType.url,
        )
        for i in range(n)
    ]


def _make_leaf(title: str) -> SkeletonNode:
    """Helper to create a leaf node (simulates a page)."""
    return SkeletonNode(
        title=title,
        ref=f"https://example.com/{title.lower().replace(' ', '-')}",
        ref_type=RefType.url,
        page=CrawledPage(
            url_or_path=f"https://example.com/{title.lower().replace(' ', '-')}",
            title=title,
            content="",
            ref_type=RefType.url,
        ),
    )


class MockProvider(LLMProvider):
    def __init__(self, text: str = "", raise_error: bool = False):
        self._text = text
        self._raise_error = raise_error

    async def generate(self, user_message, system="", max_tokens=256, temperature=0.0):
        if self._raise_error:
            raise Exception("API error")
        return LLMResponse(text=self._text)


class TestParseResponse:
    def test_simple_categories(self):
        pages = _make_pages(4)
        response = json.dumps({
            "categories": [
                {"title": "Group A", "page_indices": [0, 1]},
                {"title": "Group B", "page_indices": [2, 3]},
            ]
        })
        tree = _parse_response(response, pages)
        assert len(tree.children) == 2
        assert tree.children[0].title == "Group A"
        assert tree.count_leaves() == 4

    def test_missing_pages_go_to_other(self):
        pages = _make_pages(4)
        response = json.dumps({
            "categories": [
                {"title": "Group A", "page_indices": [0, 1]},
            ]
        })
        tree = _parse_response(response, pages)
        # Pages 2 and 3 should be in an "Other" category
        other = next((c for c in tree.children if c.title == "Other"), None)
        assert other is not None
        assert other.count_leaves() == 2

    def test_out_of_bounds_indices_ignored(self):
        pages = _make_pages(2)
        response = json.dumps({
            "categories": [
                {"title": "All", "page_indices": [0, 1, 99]},
            ]
        })
        tree = _parse_response(response, pages)
        assert tree.count_leaves() == 2

    def test_handles_code_fences(self):
        pages = _make_pages(2)
        response = '```json\n{"categories": [{"title": "All", "page_indices": [0, 1]}]}\n```'
        tree = _parse_response(response, pages)
        assert tree.count_leaves() == 2

    def test_json_with_trailing_commentary(self):
        """Handles LLM output that has extra text after the JSON."""
        pages = _make_pages(3)
        response = (
            '{"categories": [{"title": "All", "page_indices": [0, 1, 2]}]}\n\n'
            'Here are the categories I created for you.\n'
            'Let me know if you need changes.'
        )
        tree = _parse_response(response, pages)
        assert tree.count_leaves() == 3

    def test_json_with_leading_text(self):
        """Handles LLM output that has text before the JSON."""
        pages = _make_pages(2)
        response = (
            'Sure, here are the categories:\n\n'
            '{"categories": [{"title": "All", "page_indices": [0, 1]}]}'
        )
        tree = _parse_response(response, pages)
        assert tree.count_leaves() == 2

    def test_duplicate_indices_ignored(self):
        pages = _make_pages(3)
        response = json.dumps({
            "categories": [
                {"title": "A", "page_indices": [0, 1]},
                {"title": "B", "page_indices": [1, 2]},  # index 1 is duplicate
            ]
        })
        tree = _parse_response(response, pages)
        assert tree.count_leaves() == 3  # each page once


class TestIsCatchAll:
    def test_recognizes_other(self):
        assert _is_catch_all("Other") is True

    def test_recognizes_miscellaneous(self):
        assert _is_catch_all("Miscellaneous") is True

    def test_recognizes_misc(self):
        assert _is_catch_all("Misc") is True

    def test_recognizes_general(self):
        assert _is_catch_all("General") is True

    def test_recognizes_uncategorized(self):
        assert _is_catch_all("Uncategorized") is True

    def test_case_insensitive(self):
        assert _is_catch_all("OTHER") is True
        assert _is_catch_all("  other  ") is True

    def test_rejects_normal_names(self):
        assert _is_catch_all("Configuration") is False
        assert _is_catch_all("Getting Started") is False
        assert _is_catch_all("IDE Integrations") is False


class TestCollapseSingleChildren:
    def test_single_child_category_collapsed(self):
        """A category with exactly 1 child should be collapsed."""
        leaf = _make_leaf("Only Page")
        wrapper = SkeletonNode(title="Wrapper Category", children=[leaf])
        root = SkeletonNode(title="Root", children=[wrapper])

        result = _collapse_single_children(root)

        # The wrapper should be gone; the leaf is promoted
        assert len(result.children) == 1
        assert result.children[0].title == "Only Page"
        assert result.children[0].is_leaf

    def test_multi_child_category_kept(self):
        """A category with 2+ children should not be collapsed."""
        leaf1 = _make_leaf("Page A")
        leaf2 = _make_leaf("Page B")
        category = SkeletonNode(title="Good Category", children=[leaf1, leaf2])
        root = SkeletonNode(title="Root", children=[category])

        result = _collapse_single_children(root)

        assert len(result.children) == 1
        assert result.children[0].title == "Good Category"
        assert len(result.children[0].children) == 2

    def test_nested_single_child_chain_fully_collapsed(self):
        """A chain of single-child wrappers should be fully collapsed."""
        leaf = _make_leaf("Deep Page")
        inner = SkeletonNode(title="Inner Wrapper", children=[leaf])
        outer = SkeletonNode(title="Outer Wrapper", children=[inner])
        root = SkeletonNode(title="Root", children=[outer])

        result = _collapse_single_children(root)

        # Both wrappers should be gone; leaf is promoted to root
        assert len(result.children) == 1
        assert result.children[0].title == "Deep Page"
        assert result.children[0].is_leaf

    def test_leaf_node_unchanged(self):
        """A leaf node should pass through unchanged."""
        leaf = _make_leaf("Single Leaf")
        result = _collapse_single_children(leaf)
        assert result.title == "Single Leaf"
        assert result.is_leaf

    def test_mixed_categories(self):
        """Some categories collapse, others don't."""
        leaf_a = _make_leaf("Page A")
        leaf_b = _make_leaf("Page B")
        leaf_c = _make_leaf("Page C")

        # This one has 1 child -> should collapse
        single = SkeletonNode(title="Single Wrapper", children=[leaf_a])
        # This one has 2 children -> should stay
        multi = SkeletonNode(title="Multi Category", children=[leaf_b, leaf_c])
        root = SkeletonNode(title="Root", children=[single, multi])

        result = _collapse_single_children(root)

        assert len(result.children) == 2
        assert result.children[0].title == "Page A"
        assert result.children[0].is_leaf
        assert result.children[1].title == "Multi Category"
        assert len(result.children[1].children) == 2

    def test_preserves_total_leaf_count(self):
        """Collapsing should never lose or duplicate pages."""
        leaves = [_make_leaf(f"Page {i}") for i in range(5)]
        # Create a mix: two single-child wrappers, one multi-child category
        w1 = SkeletonNode(title="W1", children=[leaves[0]])
        w2 = SkeletonNode(title="W2", children=[leaves[1]])
        cat = SkeletonNode(title="Cat", children=[leaves[2], leaves[3], leaves[4]])
        root = SkeletonNode(title="Root", children=[w1, w2, cat])

        result = _collapse_single_children(root)
        assert result.count_leaves() == 5


class TestMergeCatchAllDuplicates:
    def test_merge_matching_subcategories_into_top_level(self):
        """Subcategories in 'Other' that match top-level names get merged."""
        # Top-level "IDE Integrations" with 2 pages
        ide_leaf1 = _make_leaf("VS Code Setup")
        ide_leaf2 = _make_leaf("Neovim Setup")
        ide_category = SkeletonNode(
            title="IDE Integrations",
            children=[ide_leaf1, ide_leaf2],
        )

        # "Other" contains a subcategory also called "IDE Integrations"
        other_ide_leaf = _make_leaf("JetBrains Setup")
        other_ide_sub = SkeletonNode(
            title="IDE Integrations",
            children=[other_ide_leaf],
        )
        other_misc_leaf = _make_leaf("Troubleshooting")
        other_category = SkeletonNode(
            title="Other",
            children=[other_ide_sub, other_misc_leaf],
        )

        root = SkeletonNode(title="Root", children=[ide_category, other_category])
        result = _merge_catch_all_duplicates(root)

        # "Other" should be dissolved
        titles = [c.title for c in result.children]
        assert "Other" not in titles

        # "IDE Integrations" should now have 3 pages (2 original + 1 merged)
        ide_node = next(c for c in result.children if c.title == "IDE Integrations")
        assert ide_node.count_leaves() == 3

        # "Troubleshooting" should be promoted to root level
        assert "Troubleshooting" in titles

    def test_no_catch_all_no_change(self):
        """If there's no catch-all category, tree is unchanged."""
        leaf_a = _make_leaf("Page A")
        leaf_b = _make_leaf("Page B")
        cat1 = SkeletonNode(title="Category 1", children=[leaf_a])
        cat2 = SkeletonNode(title="Category 2", children=[leaf_b])
        root = SkeletonNode(title="Root", children=[cat1, cat2])

        result = _merge_catch_all_duplicates(root)
        assert len(result.children) == 2
        assert result.children[0].title == "Category 1"
        assert result.children[1].title == "Category 2"

    def test_catch_all_with_no_matching_subcategories(self):
        """If 'Other' has subcategories that don't match top-level, dissolve and promote."""
        leaf_a = _make_leaf("Page A")
        cat1 = SkeletonNode(title="Configuration", children=[leaf_a])

        leaf_b = _make_leaf("Hooks Guide")
        leaf_c = _make_leaf("Debug Tips")
        other = SkeletonNode(title="Other", children=[leaf_b, leaf_c])

        root = SkeletonNode(title="Root", children=[cat1, other])
        result = _merge_catch_all_duplicates(root)

        # "Other" is dissolved; its children are promoted
        titles = [c.title for c in result.children]
        assert "Other" not in titles
        assert "Hooks Guide" in titles
        assert "Debug Tips" in titles

    def test_case_insensitive_matching(self):
        """Matching between catch-all subcategories and top-level is case-insensitive."""
        leaf1 = _make_leaf("VS Code")
        top = SkeletonNode(title="IDE Integrations", children=[leaf1])

        leaf2 = _make_leaf("JetBrains")
        sub = SkeletonNode(title="ide integrations", children=[leaf2])
        other = SkeletonNode(title="Miscellaneous", children=[sub])

        root = SkeletonNode(title="Root", children=[top, other])
        result = _merge_catch_all_duplicates(root)

        # Should merge into IDE Integrations
        ide_node = next(c for c in result.children if c.title == "IDE Integrations")
        assert ide_node.count_leaves() == 2
        titles = [c.title for c in result.children]
        assert "Miscellaneous" not in titles

    def test_preserves_total_leaf_count(self):
        """Merging should never lose or duplicate pages."""
        leaves = [_make_leaf(f"Page {i}") for i in range(6)]
        cat = SkeletonNode(title="Category A", children=[leaves[0], leaves[1]])
        sub_matching = SkeletonNode(title="Category A", children=[leaves[2]])
        sub_other = SkeletonNode(title="Unique Topic", children=[leaves[3], leaves[4]])
        other = SkeletonNode(
            title="Other",
            children=[sub_matching, sub_other, leaves[5]],
        )
        root = SkeletonNode(title="Root", children=[cat, other])

        result = _merge_catch_all_duplicates(root)
        assert result.count_leaves() == 6


class TestEnforceMinGroupSize:
    def test_small_subcategory_dissolved(self):
        """A subcategory with fewer than min_size leaves is dissolved."""
        leaf_a = _make_leaf("Page A")
        small_cat = SkeletonNode(title="Tiny Category", children=[leaf_a])
        leaf_b = _make_leaf("Page B")
        leaf_c = _make_leaf("Page C")
        big_cat = SkeletonNode(title="Big Category", children=[leaf_b, leaf_c])
        root = SkeletonNode(title="Root", children=[small_cat, big_cat])

        result = _enforce_min_group_size(root, min_size=2)

        # "Tiny Category" (1 leaf) should be dissolved; leaf promoted
        titles = [c.title for c in result.children]
        assert "Tiny Category" not in titles
        assert "Page A" in titles
        # "Big Category" (2 leaves) should remain
        assert "Big Category" in titles
        assert result.count_leaves() == 3

    def test_groups_at_threshold_kept(self):
        """A subcategory with exactly min_size leaves is kept."""
        leaf_a = _make_leaf("Page A")
        leaf_b = _make_leaf("Page B")
        cat = SkeletonNode(title="Just Enough", children=[leaf_a, leaf_b])
        root = SkeletonNode(title="Root", children=[cat])

        result = _enforce_min_group_size(root, min_size=2)
        assert len(result.children) == 1
        assert result.children[0].title == "Just Enough"

    def test_recursive_dissolution(self):
        """Deeply nested small groups are dissolved bottom-up."""
        leaf = _make_leaf("Deep Leaf")
        inner = SkeletonNode(title="Inner", children=[leaf])
        outer = SkeletonNode(title="Outer", children=[inner])
        root = SkeletonNode(title="Root", children=[outer])

        result = _enforce_min_group_size(root, min_size=2)
        # Both Inner and Outer should be dissolved (each has 1 leaf)
        assert len(result.children) == 1
        assert result.children[0].title == "Deep Leaf"
        assert result.children[0].is_leaf

    def test_preserves_total_leaf_count(self):
        """Dissolution should never lose or duplicate pages."""
        leaves = [_make_leaf(f"Page {i}") for i in range(5)]
        small = SkeletonNode(title="Small", children=[leaves[0]])
        big = SkeletonNode(title="Big", children=[leaves[1], leaves[2], leaves[3]])
        also_small = SkeletonNode(title="Also Small", children=[leaves[4]])
        root = SkeletonNode(title="Root", children=[small, big, also_small])

        result = _enforce_min_group_size(root, min_size=2)
        assert result.count_leaves() == 5


class TestPostprocessTree:
    def test_full_pipeline_real_world_scenario(self):
        """Simulate the Claude Code docs scenario: Other with duplicate subcategories
        and single-child wrappers."""
        # Build a tree that mimics the real issues:
        # 1. "IDE Integrations" at top level
        # 2. "Other" with a sub called "IDE Integrations" (duplicate)
        # 3. "Advanced Usage & Automation" with only 1 leaf (single-child wrapper)
        ide1 = _make_leaf("VS Code")
        ide2 = _make_leaf("Neovim")
        ide_top = SkeletonNode(title="IDE Integrations", children=[ide1, ide2])

        other_ide = _make_leaf("JetBrains")
        other_ide_sub = SkeletonNode(title="IDE Integrations", children=[other_ide])
        hooks = _make_leaf("Hooks Reference")
        troubleshooting = _make_leaf("Troubleshooting")
        other = SkeletonNode(
            title="Other",
            children=[other_ide_sub, hooks, troubleshooting],
        )

        single_leaf = _make_leaf("CI/CD Automation")
        single_wrapper = SkeletonNode(
            title="Advanced Usage & Automation",
            children=[single_leaf],
        )

        config1 = _make_leaf("Settings")
        config2 = _make_leaf("Environment Variables")
        config = SkeletonNode(title="Configuration", children=[config1, config2])

        root = SkeletonNode(
            title="Root",
            children=[ide_top, other, single_wrapper, config],
        )

        result = _postprocess_tree(root)

        # 1. "Other" should be gone
        titles = [c.title for c in result.children]
        assert "Other" not in titles

        # 2. "IDE Integrations" should have 3 pages (merged)
        ide_node = next(c for c in result.children if c.title == "IDE Integrations")
        assert ide_node.count_leaves() == 3

        # 3. "Advanced Usage & Automation" wrapper should be collapsed
        assert "Advanced Usage & Automation" not in titles

        # 4. Total leaf count preserved (8 pages total)
        assert result.count_leaves() == 8

    def test_postprocess_on_leaf_is_noop(self):
        """Postprocessing a leaf node should return it unchanged."""
        leaf = _make_leaf("Single Page")
        result = _postprocess_tree(leaf)
        assert result.title == "Single Page"
        assert result.is_leaf


class TestLLMStructurer:
    @pytest.mark.asyncio
    async def test_with_mocked_provider(self):
        pages = _make_pages(10)
        response_text = json.dumps({
            "categories": [
                {"title": "Group A", "page_indices": [0, 1, 2, 3, 4]},
                {"title": "Group B", "page_indices": [5, 6, 7, 8, 9]},
            ]
        })
        provider = MockProvider(text=response_text)

        structurer = LLMStructurer(provider=provider)
        tree = await structurer.structure(pages)

        assert len(tree.children) == 2
        assert tree.count_leaves() == 10

    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        pages = _make_pages(10)
        provider = MockProvider(raise_error=True)

        structurer = LLMStructurer(provider=provider)
        tree = await structurer.structure(pages)

        # Should fall back to flat tree
        assert tree.count_leaves() == 10
        assert len(tree.children) == 10  # All pages as direct children

    @pytest.mark.asyncio
    async def test_empty_pages(self):
        provider = MockProvider()
        structurer = LLMStructurer(provider=provider)
        tree = await structurer.structure([])
        assert tree.title == "Root"
        assert tree.is_leaf

    @pytest.mark.asyncio
    async def test_small_set_returns_flat(self):
        """Pages at or below the leaf threshold should not call the LLM."""
        pages = _make_pages(5)
        provider = MockProvider(raise_error=True)  # Would fail if called

        structurer = LLMStructurer(provider=provider, leaf_threshold=7)
        tree = await structurer.structure(pages)

        assert tree.count_leaves() == 5
        assert len(tree.children) == 5

    @pytest.mark.asyncio
    async def test_recursive_subgrouping(self):
        """Categories larger than leaf_threshold should be recursively split."""
        pages = _make_pages(20)
        call_count = 0

        class RecursiveMockProvider(LLMProvider):
            async def generate(self, user_message, system="", max_tokens=256, temperature=0.0):
                nonlocal call_count
                call_count += 1
                # Parse how many pages we're grouping
                lines = [l for l in user_message.split("\n") if l.startswith("[")]
                n = len(lines)
                # Split roughly in half
                mid = n // 2
                return LLMResponse(text=json.dumps({
                    "categories": [
                        {"title": f"Half A (depth {call_count})", "page_indices": list(range(mid))},
                        {"title": f"Half B (depth {call_count})", "page_indices": list(range(mid, n))},
                    ]
                }))

        structurer = LLMStructurer(
            provider=RecursiveMockProvider(),
            leaf_threshold=6,
        )
        tree = await structurer.structure(pages)

        # Should have made multiple LLM calls (at least the top-level split
        # plus recursive splits for the two halves of 10)
        assert call_count >= 3
        assert tree.count_leaves() == 20
        # Tree should have depth > 1 (categories within categories)
        assert any(not c.is_leaf and any(not gc.is_leaf for gc in c.children) for c in tree.children)

    @pytest.mark.asyncio
    async def test_recursive_pass_uses_parent_context(self):
        """Recursive sub-grouping should pass parent_category to the prompt."""
        pages = _make_pages(16)
        captured_systems: list[str] = []

        class CapturingProvider(LLMProvider):
            async def generate(self, user_message, system="", max_tokens=256, temperature=0.0):
                captured_systems.append(system)
                lines = [l for l in user_message.split("\n") if l.startswith("[")]
                n = len(lines)
                mid = n // 2
                return LLMResponse(text=json.dumps({
                    "categories": [
                        {"title": "Sub A", "page_indices": list(range(mid))},
                        {"title": "Sub B", "page_indices": list(range(mid, n))},
                    ]
                }))

        structurer = LLMStructurer(
            provider=CapturingProvider(),
            leaf_threshold=5,
        )
        await structurer.structure(pages)

        # First call (depth 0) should use the base prompt (no parent_category)
        assert "sub-grouping" not in captured_systems[0].lower()

        # Recursive calls (depth > 0) should mention sub-grouping and parent
        recursive_calls = captured_systems[1:]
        assert len(recursive_calls) > 0
        for sys_prompt in recursive_calls:
            assert "sub-grouping" in sys_prompt.lower() or "Sub-Category" in sys_prompt

    @pytest.mark.asyncio
    async def test_postprocessing_applied_after_structuring(self):
        """The structure() method should apply postprocessing (e.g. collapse single children)."""
        pages = _make_pages(10)
        # LLM returns a group with only 1 page -> should be collapsed by postprocessing
        response_text = json.dumps({
            "categories": [
                {"title": "Big Group", "page_indices": [0, 1, 2, 3, 4, 5, 6, 7]},
                {"title": "Lonely Wrapper", "page_indices": [8]},
                {"title": "Another Group", "page_indices": [9]},
            ]
        })
        provider = MockProvider(text=response_text)
        structurer = LLMStructurer(provider=provider)
        tree = await structurer.structure(pages)

        # "Lonely Wrapper" had 1 child -> should be collapsed
        # "Another Group" also had 1 child -> should be collapsed
        # Both are dissolved by min-group-size or single-child collapse
        titles = [c.title for c in tree.children]
        assert "Lonely Wrapper" not in titles
        assert "Another Group" not in titles
        assert tree.count_leaves() == 10
