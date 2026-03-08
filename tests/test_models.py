"""Tests for core data models."""

import json
from pathlib import Path

import pytest

from sema_tree.models import SemaTree, IndexNode, RefType, Source, SourceType


def _sample_tree() -> IndexNode:
    """Build a small sample tree for testing."""
    return IndexNode(
        id="0",
        title="Root",
        summary="Root summary",
        children=[
            IndexNode(
                id="0.0",
                title="Section A",
                summary="Section A summary",
                source_id="src1",
                children=[
                    IndexNode(
                        id="0.0.0",
                        title="Page 1",
                        summary="Page 1 summary",
                        ref="https://example.com/page1",
                        ref_type=RefType.url,
                        source_id="src1",
                    ),
                    IndexNode(
                        id="0.0.1",
                        title="Page 2",
                        summary="Page 2 summary",
                        ref="https://example.com/page2",
                        ref_type=RefType.url,
                        source_id="src1",
                    ),
                ],
            ),
            IndexNode(
                id="0.1",
                title="Section B",
                summary="Section B summary",
                source_id="src1",
                children=[
                    IndexNode(
                        id="0.1.0",
                        title="Page 3",
                        summary="Page 3 summary",
                        ref="/docs/page3.md",
                        ref_type=RefType.file,
                        source_id="src1",
                    ),
                ],
            ),
        ],
    )


class TestIndexNode:
    def test_is_leaf(self):
        leaf = IndexNode(id="0.0", title="Leaf", summary="A leaf")
        assert leaf.is_leaf is True

        tree = _sample_tree()
        assert tree.is_leaf is False

    def test_depth(self):
        tree = _sample_tree()
        assert tree.depth == 2

        leaf = IndexNode(id="0", title="Leaf", summary="")
        assert leaf.depth == 0

    def test_count_leaves(self):
        tree = _sample_tree()
        assert tree.count_leaves() == 3

    def test_count_nodes(self):
        tree = _sample_tree()
        assert tree.count_nodes() == 6  # root + 2 sections + 3 pages

    def test_find_node(self):
        tree = _sample_tree()
        assert tree.find_node("0") is tree
        assert tree.find_node("0.0.1").title == "Page 2"
        assert tree.find_node("0.1.0").title == "Page 3"
        assert tree.find_node("nonexistent") is None

    def test_all_leaves(self):
        tree = _sample_tree()
        leaves = tree.all_leaves()
        assert len(leaves) == 3
        titles = {l.title for l in leaves}
        assert titles == {"Page 1", "Page 2", "Page 3"}

    def test_serialization_roundtrip(self):
        tree = _sample_tree()
        data = tree.model_dump()
        restored = IndexNode.model_validate(data)
        assert restored.id == tree.id
        assert restored.depth == tree.depth
        assert restored.count_nodes() == tree.count_nodes()

    def test_json_roundtrip(self):
        tree = _sample_tree()
        json_str = tree.model_dump_json()
        restored = IndexNode.model_validate_json(json_str)
        assert restored.find_node("0.1.0").ref == "/docs/page3.md"


class TestSource:
    def test_creation(self):
        source = Source(
            id="test-src",
            type=SourceType.website,
            origin="https://example.com",
            page_count=10,
        )
        assert source.id == "test-src"
        assert source.type == SourceType.website
        assert source.page_count == 10
        assert source.crawled_at is not None


class TestSemaTree:
    def test_default_creation(self):
        index = SemaTree()
        assert index.version == "1.0"
        assert index.root.id == "0"
        assert len(index.sources) == 0

    def test_find_source(self):
        index = SemaTree(
            sources=[
                Source(id="s1", type=SourceType.website, origin="https://a.com"),
                Source(id="s2", type=SourceType.local_folder, origin="/docs"),
            ]
        )
        assert index.find_source("s1").origin == "https://a.com"
        assert index.find_source("s2").type == SourceType.local_folder
        assert index.find_source("s3") is None

    def test_save_and_load(self, tmp_path):
        index = SemaTree()
        index.root = _sample_tree()
        index.sources = [
            Source(id="test", type=SourceType.website, origin="https://example.com", page_count=3)
        ]

        path = str(tmp_path / "test_index.json")
        index.save(path)

        loaded = SemaTree.load(path)
        assert loaded.version == "1.0"
        assert loaded.root.count_leaves() == 3
        assert len(loaded.sources) == 1
        assert loaded.sources[0].id == "test"

    def test_load_fixture(self):
        fixture_path = str(Path(__file__).parent / "fixtures" / "sample_index.json")
        index = SemaTree.load(fixture_path)
        assert index.version == "1.0"
        assert len(index.sources) == 1
        assert index.root.count_leaves() == 4  # getting-started, auth, endpoints, config
        assert index.root.find_node("0.0.1.0").title == "Authentication"
