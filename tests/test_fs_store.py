"""Tests for FileSystemStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_index.fs_store import FileSystemStore, _clean_filename
from agentic_index.models import AgenticIndex, IndexNode, RefType, Source, SourceType


def _make_index() -> AgenticIndex:
    """Build a minimal AgenticIndex for testing persistence."""
    return AgenticIndex(
        sources=[
            Source(id="test-src", type=SourceType.website, origin="https://example.com", page_count=3)
        ],
        root=IndexNode(
            id="0",
            title="Root",
            summary="Root summary",
            children=[
                IndexNode(
                    id="0.0",
                    title="Getting Started",
                    summary="How to get started with the tool.",
                    nav_summary="Install and configure the tool",
                    source_id="test-src",
                    children=[
                        IndexNode(
                            id="0.0.0",
                            title="Installation",
                            summary="Step-by-step installation guide.",
                            nav_summary="Install via pip or uv",
                            ref="https://example.com/install",
                            ref_type=RefType.url,
                            source_id="test-src",
                        ),
                        IndexNode(
                            id="0.0.1",
                            title="Quick Start",
                            summary="Get up and running in minutes.",
                            nav_summary="Run your first command",
                            ref="https://example.com/quickstart",
                            ref_type=RefType.url,
                            source_id="test-src",
                        ),
                    ],
                ),
                IndexNode(
                    id="0.1",
                    title="API Reference",
                    summary="Full API documentation.",
                    nav_summary="All public API methods",
                    ref="https://example.com/api",
                    ref_type=RefType.url,
                    source_id="test-src",
                ),
            ],
        ),
    )


class TestCleanFilename:
    def test_basic(self):
        assert _clean_filename("Getting Started") == "getting-started"

    def test_removes_special_chars(self):
        assert _clean_filename("API & Reference!") == "api-reference"

    def test_collapses_whitespace(self):
        assert _clean_filename("  Hello   World  ") == "hello-world"

    def test_empty_fallback(self):
        assert _clean_filename("!!!") == "untitled"

    def test_underscores_become_dashes(self):
        assert _clean_filename("my_module") == "my-module"


class TestFileSystemStoreSave:
    def test_creates_root_directory(self, tmp_path):
        store = FileSystemStore(tmp_path / "index")
        index = _make_index()
        store.save(index)
        assert store.root.exists()

    def test_creates_meta_json(self, tmp_path):
        store = FileSystemStore(tmp_path / "index")
        store.save(_make_index())
        assert (store.root / "_meta.json").exists()

    def test_creates_group_directories(self, tmp_path):
        store = FileSystemStore(tmp_path / "index")
        store.save(_make_index())
        # "Getting Started" -> getting-started/
        assert (store.root / "getting-started").is_dir()

    def test_creates_leaf_files(self, tmp_path):
        store = FileSystemStore(tmp_path / "index")
        store.save(_make_index())
        # Leaf "Installation" inside "Getting Started"
        assert (store.root / "getting-started" / "installation.md").is_file()
        assert (store.root / "getting-started" / "quick-start.md").is_file()

    def test_leaf_file_has_frontmatter(self, tmp_path):
        store = FileSystemStore(tmp_path / "index")
        store.save(_make_index())
        content = (store.root / "getting-started" / "installation.md").read_text()
        assert "---" in content
        assert "ref: https://example.com/install" in content
        assert "ref_type: url" in content
        assert "Installation" in content

    def test_group_directory_has_summary(self, tmp_path):
        store = FileSystemStore(tmp_path / "index")
        store.save(_make_index())
        summary_file = store.root / "getting-started" / "_summary.md"
        assert summary_file.exists()
        content = summary_file.read_text()
        assert "How to get started" in content

    def test_skips_root_node_level(self, tmp_path):
        store = FileSystemStore(tmp_path / "index")
        store.save(_make_index())
        # Root node (id="0", title="Root") should be skipped — no "root/" subdir
        assert not (store.root / "root").exists()

    def test_top_level_leaf_as_file(self, tmp_path):
        store = FileSystemStore(tmp_path / "index")
        store.save(_make_index())
        # "API Reference" is a top-level leaf -> api-reference.md at root level
        assert (store.root / "api-reference.md").is_file()


class TestFileSystemStoreListDir:
    @pytest.fixture
    def populated_store(self, tmp_path) -> FileSystemStore:
        store = FileSystemStore(tmp_path / "index")
        store.save(_make_index())
        return store

    def test_list_root(self, populated_store):
        result = populated_store.list_dir("/")
        assert "getting-started" in result
        assert "api-reference" in result

    def test_list_subdirectory(self, populated_store):
        result = populated_store.list_dir("/getting-started")
        assert "installation" in result.lower()
        assert "quick-start" in result.lower()

    def test_include_summaries(self, populated_store):
        result = populated_store.list_dir("/", include_summaries=True)
        assert "Install and configure" in result

    def test_nonexistent_path(self, populated_store):
        result = populated_store.list_dir("/does-not-exist")
        assert "Error" in result

    def test_depth_controls_recursion(self, populated_store):
        shallow = populated_store.list_dir("/", depth=1)
        deep = populated_store.list_dir("/", depth=2)
        # Deeper listing should have more items
        assert len(deep) >= len(shallow)


class TestFileSystemStoreReadFile:
    @pytest.fixture
    def populated_store(self, tmp_path) -> FileSystemStore:
        store = FileSystemStore(tmp_path / "index")
        store.save(_make_index())
        return store

    def test_read_leaf_file(self, populated_store):
        content = populated_store.read_file("/getting-started/installation.md")
        assert "Installation" in content
        assert "https://example.com/install" in content

    def test_read_file_without_extension(self, populated_store):
        # Should auto-append .md
        content = populated_store.read_file("/getting-started/installation")
        assert "Installation" in content

    def test_read_directory_returns_summary(self, populated_store):
        content = populated_store.read_file("/getting-started")
        assert "How to get started" in content

    def test_read_nonexistent_returns_error(self, populated_store):
        result = populated_store.read_file("/nonexistent/path.md")
        assert "Error" in result


class TestFileSystemStoreFind:
    @pytest.fixture
    def populated_store(self, tmp_path) -> FileSystemStore:
        store = FileSystemStore(tmp_path / "index")
        store.save(_make_index())
        return store

    def test_find_by_substring(self, populated_store):
        result = populated_store.find("install")
        assert "installation" in result.lower()

    def test_find_by_glob(self, populated_store):
        result = populated_store.find("*.md")
        assert "installation.md" in result or "api-reference.md" in result

    def test_find_no_matches(self, populated_store):
        result = populated_store.find("xyznonexistent")
        assert "No matches found" in result

    def test_find_is_case_insensitive(self, populated_store):
        lower = populated_store.find("installation")
        upper = populated_store.find("INSTALLATION")
        # Both should find the file
        assert "installation" in lower.lower()
        assert "installation" in upper.lower()

    def test_find_includes_directories(self, populated_store):
        result = populated_store.find("getting-started")
        assert "getting-started" in result
