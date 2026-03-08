"""Tests for keyword search."""

from sema_tree.models import SemaTree, IndexNode, RefType, Source, SourceType
from sema_tree.search import SearchResult, search_index


def _make_index() -> SemaTree:
    """Build a test index with known content."""
    return SemaTree(
        sources=[
            Source(id="test", type=SourceType.website, origin="https://example.com")
        ],
        root=IndexNode(
            id="0",
            title="Root",
            summary="Documentation index",
            children=[
                IndexNode(
                    id="0.0",
                    title="Getting Started",
                    summary="Installation and setup guide for beginners",
                    ref="https://example.com/getting-started",
                    ref_type=RefType.url,
                    source_id="test",
                ),
                IndexNode(
                    id="0.1",
                    title="API Reference",
                    summary="Complete REST API documentation with authentication",
                    ref="https://example.com/api",
                    ref_type=RefType.url,
                    source_id="test",
                ),
                IndexNode(
                    id="0.2",
                    title="Authentication Guide",
                    summary="How to authenticate with OAuth and API keys",
                    ref="https://example.com/auth",
                    ref_type=RefType.url,
                    source_id="test",
                ),
                IndexNode(
                    id="0.3",
                    title="Configuration",
                    summary="Environment variables and config file options",
                    ref="https://example.com/config",
                    ref_type=RefType.url,
                    source_id="test",
                ),
            ],
        ),
    )


class TestSearchIndex:
    def test_basic_search(self):
        index = _make_index()
        results = search_index(index, "authentication")
        assert len(results) > 0
        # Authentication Guide should rank high (in title)
        assert results[0].title == "Authentication Guide"

    def test_case_insensitive(self):
        index = _make_index()
        results_lower = search_index(index, "api")
        results_upper = search_index(index, "API")
        assert len(results_lower) == len(results_upper)

    def test_max_results(self):
        index = _make_index()
        results = search_index(index, "documentation", max_results=2)
        assert len(results) <= 2

    def test_empty_query(self):
        index = _make_index()
        results = search_index(index, "")
        assert results == []

    def test_no_matches(self):
        index = _make_index()
        results = search_index(index, "xyznonexistent")
        assert results == []

    def test_multi_word_query(self):
        index = _make_index()
        results = search_index(index, "API authentication")
        assert len(results) > 0
        # Both API Reference and Authentication Guide should appear
        titles = {r.title for r in results}
        assert "API Reference" in titles
        assert "Authentication Guide" in titles

    def test_title_boost(self):
        index = _make_index()
        # "configuration" appears in title of one node and summary of none directly
        results = search_index(index, "configuration")
        assert len(results) > 0
        assert results[0].node_id == "0.3"

    def test_result_fields(self):
        index = _make_index()
        results = search_index(index, "getting started")
        assert len(results) > 0
        r = results[0]
        assert isinstance(r, SearchResult)
        assert r.node_id is not None
        assert r.title is not None
        assert r.score > 0
