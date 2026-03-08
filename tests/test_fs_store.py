"""Tests for FileSystemStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_index.fs_store import FileSystemStore, _clean_filename, _parse_fm_field
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


class TestParseFmField:
    def test_plain_value(self):
        fm = "id: 0.0.1\nref: https://example.com"
        assert _parse_fm_field(fm, "id") == "0.0.1"

    def test_quoted_value(self):
        fm = 'title: "My Title"'
        assert _parse_fm_field(fm, "title", unquote=True) == "My Title"

    def test_missing_field(self):
        assert _parse_fm_field("id: 1", "title") == ""

    def test_url_field_with_colon(self):
        fm = "ref: https://example.com/path"
        assert _parse_fm_field(fm, "ref") == "https://example.com/path"


class TestFileSystemStoreLoad:
    """Round-trip tests: save() → load() produces equivalent AgenticIndex."""

    @pytest.fixture
    def saved_store(self, tmp_path) -> tuple[FileSystemStore, AgenticIndex]:
        store = FileSystemStore(tmp_path / "index")
        index = _make_index()
        store.save(index)
        return store, index

    def test_load_returns_agenticindex(self, saved_store):
        store, _ = saved_store
        loaded = FileSystemStore.load(store.root)
        assert isinstance(loaded, AgenticIndex)

    def test_load_preserves_version(self, saved_store):
        store, original = saved_store
        loaded = FileSystemStore.load(store.root)
        assert loaded.version == original.version

    def test_load_preserves_sources(self, saved_store):
        store, original = saved_store
        loaded = FileSystemStore.load(store.root)
        assert len(loaded.sources) == len(original.sources)
        assert loaded.sources[0].id == "test-src"
        assert loaded.sources[0].origin == "https://example.com"
        assert loaded.sources[0].page_count == 3

    def test_load_preserves_source_type(self, saved_store):
        store, _ = saved_store
        loaded = FileSystemStore.load(store.root)
        assert loaded.sources[0].type == SourceType.website

    def test_load_reconstructs_root(self, saved_store):
        store, _ = saved_store
        loaded = FileSystemStore.load(store.root)
        assert loaded.root.id == "0"
        assert loaded.root.title == "Root"

    def test_load_reconstructs_leaf_ref(self, saved_store):
        store, _ = saved_store
        loaded = FileSystemStore.load(store.root)
        leaves = loaded.root.all_leaves()
        refs = {leaf.ref for leaf in leaves}
        assert "https://example.com/install" in refs
        assert "https://example.com/quickstart" in refs

    def test_load_reconstructs_leaf_ref_type(self, saved_store):
        store, _ = saved_store
        loaded = FileSystemStore.load(store.root)
        leaves = loaded.root.all_leaves()
        for leaf in leaves:
            if leaf.ref:
                assert leaf.ref_type == RefType.url

    def test_load_reconstructs_leaf_source_id(self, saved_store):
        store, _ = saved_store
        loaded = FileSystemStore.load(store.root)
        leaves = loaded.root.all_leaves()
        for leaf in leaves:
            assert leaf.source_id == "test-src"

    def test_load_reconstructs_nav_summary(self, saved_store):
        store, _ = saved_store
        loaded = FileSystemStore.load(store.root)
        leaves = loaded.root.all_leaves()
        nav_summaries = {leaf.nav_summary for leaf in leaves}
        assert "Install via pip or uv" in nav_summaries

    def test_load_reconstructs_branch_nav_summary(self, saved_store):
        store, _ = saved_store
        loaded = FileSystemStore.load(store.root)
        # "Getting Started" branch should have its nav_summary
        branches = [c for c in loaded.root.children if not c.is_leaf]
        assert any("Install and configure" in b.nav_summary for b in branches)

    def test_load_preserves_leaf_count(self, saved_store):
        store, original = saved_store
        loaded = FileSystemStore.load(store.root)
        assert loaded.root.count_leaves() == original.root.count_leaves()

    def test_load_preserves_node_count(self, saved_store):
        store, original = saved_store
        loaded = FileSystemStore.load(store.root)
        # Node count should match (root + branches + leaves)
        assert loaded.root.count_nodes() == original.root.count_nodes()

    def test_load_meta_json_sources_serialization(self, tmp_path):
        """_meta.json must contain the sources list."""
        import json

        store = FileSystemStore(tmp_path / "index")
        index = _make_index()
        store.save(index)

        meta = json.loads((store.root / "_meta.json").read_text())
        assert "sources" in meta
        assert len(meta["sources"]) == 1
        src = meta["sources"][0]
        assert src["id"] == "test-src"
        assert src["type"] == "website"
        assert src["origin"] == "https://example.com"
        assert src["page_count"] == 3

    def test_load_missing_meta_raises(self, tmp_path):
        store = FileSystemStore(tmp_path / "no_index")
        with pytest.raises(FileNotFoundError):
            FileSystemStore.load(store.root)

    def test_roundtrip_local_file_ref_type(self, tmp_path):
        """Leaf with ref_type=file round-trips correctly."""
        store = FileSystemStore(tmp_path / "local_index")
        index = AgenticIndex(
            sources=[Source(id="local-src", type=SourceType.local_folder, origin="/docs")],
            root=IndexNode(
                id="0",
                title="Root",
                children=[
                    IndexNode(
                        id="0.0",
                        title="README",
                        summary="The readme.",
                        nav_summary="Readme file",
                        ref="/docs/README.md",
                        ref_type=RefType.file,
                        source_id="local-src",
                    )
                ],
            ),
        )
        store.save(index)
        loaded = FileSystemStore.load(store.root)
        leaf = loaded.root.all_leaves()[0]
        assert leaf.ref_type == RefType.file
        assert leaf.ref == "/docs/README.md"
        assert leaf.source_id == "local-src"

    def test_roundtrip_content_hash(self, tmp_path):
        """content_hash field round-trips through save/load."""
        store = FileSystemStore(tmp_path / "hash_index")
        index = AgenticIndex(
            sources=[Source(id="s", type=SourceType.website, origin="https://x.com")],
            root=IndexNode(
                id="0",
                title="Root",
                children=[
                    IndexNode(
                        id="0.0",
                        title="Doc",
                        summary="A doc.",
                        nav_summary="A doc",
                        ref="https://x.com/doc",
                        ref_type=RefType.url,
                        source_id="s",
                        content_hash="abc123",
                    )
                ],
            ),
        )
        store.save(index)
        loaded = FileSystemStore.load(store.root)
        leaf = loaded.root.all_leaves()[0]
        assert leaf.content_hash == "abc123"
