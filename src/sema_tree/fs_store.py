"""Filesystem persistence for SemaTree.

This module handles mapping the in-memory IndexNode tree to a physical
directory structure on disk, enabling version control and human readability.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from sema_tree.models import SemaTree, IndexNode, RefType, Source, SourceType

logger = logging.getLogger(__name__)


def _clean_filename(text: str) -> str:
    """Convert text to a valid, clean filename/slug."""
    # Lowercase and replace spaces with dashes
    text = text.lower().strip()
    text = re.sub(r"[\s_]+", "-", text)
    # Remove characters that aren't alphanumerics or dashes
    text = re.sub(r"[^a-z0-9-]", "", text)
    # Remove duplicate dashes
    text = re.sub(r"-+", "-", text)
    return text.strip("-") or "untitled"


def _parse_datetime(s: str) -> datetime:
    """Parse an ISO 8601 datetime string, handling missing timezone info."""
    if not s:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def _parse_fm_field(fm_text: str, field: str, unquote: bool = False) -> str:
    """Parse a single YAML frontmatter field value."""
    match = re.search(rf"^{field}:\s*(.*)$", fm_text, re.MULTILINE)
    if not match:
        return ""
    value = match.group(1).strip()
    if unquote and value.startswith('"') and value.endswith('"'):
        value = value[1:-1].replace('\\"', '"')
    return value


def _extract_summary_from_body(body: str) -> str:
    """Extract the summary from a leaf file body (strips h1 heading and back-link)."""
    lines = body.split("\n")
    result = []
    for line in lines:
        if line.startswith("# "):
            continue
        if line.startswith("[Link to original]"):
            continue
        result.append(line)
    return "\n".join(result).strip()


class FileSystemStore:
    """Manages the filesystem representation of an SemaTree."""

    SUMMARY_FILENAME = "_summary.md"
    META_FILENAME = "_meta.json"

    def __init__(self, root_path: str | Path):
        self.root = Path(root_path)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, index: SemaTree) -> None:
        """Save the entire SemaTree to the filesystem."""
        self.root.mkdir(parents=True, exist_ok=True)

        # Save global metadata including sources
        sources_data = [
            {
                "id": s.id,
                "type": s.type.value,
                "origin": s.origin,
                "crawled_at": s.crawled_at.isoformat(),
                "page_count": s.page_count,
            }
            for s in index.sources
        ]
        index_meta = {
            "version": index.version,
            "created_at": index.created_at.isoformat(),
            "updated_at": index.updated_at.isoformat(),
            "sources": sources_data,
        }
        (self.root / self.META_FILENAME).write_text(json.dumps(index_meta, indent=2))

        self._save_node(index.root, self.root)

    def _save_node(self, node: IndexNode, current_path: Path) -> None:
        """Recursively save a node, skipping redundant 'Root' levels."""
        skip_level = (node.id == "0") or (node.title.lower() == "root")

        if skip_level:
            target_path = current_path
        else:
            slug = _clean_filename(node.title)
            target_path = current_path / slug

        if node.is_leaf:
            # Leaf node → Markdown file
            file_path = target_path.with_suffix(".md")

            ref_str = node.ref if node.ref else ""
            ref_type_str = node.ref_type.value if node.ref_type else ""
            clean_nav = re.sub(r"\s+", " ", node.nav_summary).strip().replace('"', '\\"')
            source_id_str = node.source_id or ""
            content_hash_str = node.content_hash or ""

            content = f"""---
id: {node.id}
title: "{node.title}"
nav_summary: "{clean_nav}"
ref: {ref_str}
ref_type: {ref_type_str}
source_id: {source_id_str}
content_hash: {content_hash_str}
---

# {node.title}

{node.summary}

[Link to original]({ref_str})
"""
            file_path.write_text(content, encoding="utf-8")

        else:
            # Group node → Directory
            if not skip_level:
                target_path.mkdir(exist_ok=True)

            # Always write _summary.md for branch nodes (enables load())
            if not skip_level:
                nav_comment = f"<!-- nav_summary: {node.nav_summary} -->\n" if node.nav_summary else ""
                meta = {"id": node.id, "title": node.title, "source_id": node.source_id or ""}
                meta_comment = f"<!-- meta: {json.dumps(meta)} -->\n"
                (target_path / self.SUMMARY_FILENAME).write_text(
                    nav_comment + meta_comment + (node.summary or ""),
                    encoding="utf-8",
                )

            # Recurse
            for child in node.children:
                self._save_node(child, target_path)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, root_path: str | Path) -> SemaTree:
        """Reconstruct an SemaTree from a filesystem directory.

        Reads ``_meta.json`` for version/dates/sources and walks the
        directory tree to rebuild the ``IndexNode`` hierarchy.
        """
        root = Path(root_path)

        meta_file = root / cls.META_FILENAME
        if not meta_file.exists():
            raise FileNotFoundError(f"No {cls.META_FILENAME} found in {root}")

        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        version = meta.get("version", "1.0")
        created_at = _parse_datetime(meta.get("created_at", ""))
        updated_at = _parse_datetime(meta.get("updated_at", ""))

        sources: list[Source] = []
        for s in meta.get("sources", []):
            sources.append(
                Source(
                    id=s["id"],
                    type=SourceType(s["type"]),
                    origin=s["origin"],
                    crawled_at=_parse_datetime(s.get("crawled_at", "")),
                    page_count=s.get("page_count", 0),
                )
            )

        children = cls._load_directory(root)
        root_node = IndexNode(id="0", title="Root", summary="", children=children)

        return SemaTree(
            version=version,
            created_at=created_at,
            updated_at=updated_at,
            sources=sources,
            root=root_node,
        )

    @classmethod
    def _load_directory(cls, dir_path: Path) -> list[IndexNode]:
        """Load all children from a directory, sorted by node ID."""
        children: list[IndexNode] = []

        for p in sorted(dir_path.iterdir()):
            if p.name.startswith("_"):
                continue
            if p.is_dir():
                node = cls._load_branch(p)
                if node is not None:
                    children.append(node)
            elif p.is_file() and p.suffix == ".md":
                node = cls._load_leaf(p)
                if node is not None:
                    children.append(node)

        # Sort by node ID to preserve original tree order
        def _id_sort_key(n: IndexNode) -> list[int]:
            try:
                return [int(x) for x in n.id.split(".")]
            except (ValueError, AttributeError):
                return [999]

        children.sort(key=_id_sort_key)
        return children

    @classmethod
    def _load_branch(cls, dir_path: Path) -> IndexNode | None:
        """Load a branch IndexNode from a directory."""
        summary_file = dir_path / cls.SUMMARY_FILENAME

        node_id: str | None = None
        title: str = dir_path.name.replace("-", " ").title()
        nav_summary = ""
        source_id: str | None = None
        summary = ""

        if summary_file.exists():
            content = summary_file.read_text(encoding="utf-8")

            nav_match = re.search(r"<!-- nav_summary: (.*?) -->", content)
            if nav_match:
                nav_summary = nav_match.group(1)

            meta_match = re.search(r"<!-- meta: (\{.*?\}) -->", content, re.DOTALL)
            if meta_match:
                try:
                    m = json.loads(meta_match.group(1))
                    node_id = m.get("id") or None
                    title = m.get("title") or title
                    source_id = m.get("source_id") or None
                except json.JSONDecodeError:
                    pass

            # Body after stripping HTML comments is the summary
            body = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL).strip()
            summary = body

        children = cls._load_directory(dir_path)

        return IndexNode(
            id=node_id or "0",
            title=title,
            summary=summary,
            nav_summary=nav_summary,
            source_id=source_id,
            children=children,
        )

    @classmethod
    def _load_leaf(cls, file_path: Path) -> IndexNode | None:
        """Load a leaf IndexNode from a .md file with YAML frontmatter."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        fm_match = re.match(r"^---\n(.*?)\n---\n?", content, re.DOTALL)
        if not fm_match:
            return None

        fm_text = fm_match.group(1)
        body = content[fm_match.end():]

        node_id = _parse_fm_field(fm_text, "id")
        if not node_id:
            return None

        title = _parse_fm_field(fm_text, "title", unquote=True)
        nav_summary = _parse_fm_field(fm_text, "nav_summary", unquote=True)
        ref_str = _parse_fm_field(fm_text, "ref")
        ref_type_str = _parse_fm_field(fm_text, "ref_type")
        source_id_str = _parse_fm_field(fm_text, "source_id")
        content_hash_str = _parse_fm_field(fm_text, "content_hash")

        ref = ref_str if ref_str else None
        ref_type = RefType(ref_type_str) if ref_type_str else None
        source_id = source_id_str if source_id_str else None
        content_hash = content_hash_str if content_hash_str else None

        summary = _extract_summary_from_body(body)

        return IndexNode(
            id=node_id,
            title=title or file_path.stem.replace("-", " ").title(),
            summary=summary,
            nav_summary=nav_summary,
            ref=ref,
            ref_type=ref_type,
            source_id=source_id,
            content_hash=content_hash,
        )

    # ------------------------------------------------------------------
    # Browse helpers (unchanged)
    # ------------------------------------------------------------------

    def list_dir(self, path_str: str = ".", depth: int = 1, include_summaries: bool = False) -> str:
        """List contents of a directory in the index, potentially recursively."""
        path_str = path_str.strip("/")
        if not path_str or path_str == ".":
            target = self.root
        else:
            target = self.root / path_str

        if not target.exists():
            return f"Error: Path '/{path_str}' does not exist."

        if target.is_file():
            return f"File: {target.name} (Use read() to see content)"

        lines = [f"📁 **/{path_str}**\n"]
        self._build_tree(target, lines, max_depth=depth, current_depth=0, base_path=target, include_summaries=include_summaries)
        return "\n".join(lines)

    def _build_tree(self, current_dir: Path, lines: list[str], max_depth: int, current_depth: int, base_path: Path, include_summaries: bool = False):
        """Recursively build a tree representation of the directory."""
        if current_depth >= max_depth:
            return

        children = sorted(current_dir.iterdir())

        dirs = []
        files = []

        for p in children:
            if p.name.startswith("_"):
                continue
            if p.is_dir():
                dirs.append(p)
            elif p.suffix == ".md":
                files.append(p)

        indent = "  " * current_depth

        if dirs:
            if current_depth == 0:
                lines.append("### Subdirectories:")
            for d in dirs:
                suffix = ""
                if include_summaries:
                    s_file = d / self.SUMMARY_FILENAME
                    if s_file.exists():
                        try:
                            content = s_file.read_text(encoding="utf-8")
                            match = re.search(r"<!-- nav_summary: (.*?) -->", content)
                            if match:
                                suffix = f" — {match.group(1)}"
                        except Exception:
                            pass
                lines.append(f"{indent}- **{d.name}/**{suffix}")
                self._build_tree(d, lines, max_depth, current_depth + 1, base_path, include_summaries=include_summaries)

        if files:
            if current_depth == 0:
                if dirs:
                    lines.append("")  # Spacer
                lines.append("### Documents:")
            for f in files:
                suffix = ""
                if include_summaries:
                    try:
                        content = f.read_text(encoding="utf-8")
                        match = re.search(r'^nav_summary:\s*"(.*?)"', content, re.MULTILINE)
                        if match:
                            suffix = f" — {match.group(1)}"
                    except Exception:
                        pass
                lines.append(f"{indent}- **{f.name}**{suffix}")

    def read_file(self, path_str: str) -> str:
        """Read a specific file from the index."""
        path_str = path_str.strip("/")
        target = self.root / path_str

        if target.is_dir():
            s_file = target / self.SUMMARY_FILENAME
            if s_file.exists():
                return s_file.read_text()
            return self.list_dir(path_str)

        if not target.exists():
            if not target.suffix:
                target = target.with_suffix(".md")

        if not target.exists():
            logger.error("File not found in index: %s (looked at %s)", path_str, target.absolute())
            return f"Error: File '/{path_str}' not found."

        return target.read_text(encoding="utf-8")

    def find(self, pattern: str) -> str:
        """Search for files and directories matching a pattern, grouped by parent."""
        import fnmatch
        matches: dict[str, list[str]] = {}
        pattern_lower = pattern.lower()
        has_wildcards = any(char in pattern for char in "*?[]")

        for p in self.root.rglob("*"):
            if p.name.startswith("_"):
                continue

            rel_path = p.relative_to(self.root)
            rel_path_str = str(rel_path)
            rel_path_lower = rel_path_str.lower()

            is_match = False
            if has_wildcards:
                if fnmatch.fnmatch(rel_path_lower, pattern_lower) or fnmatch.fnmatch(p.name.lower(), pattern_lower):
                    is_match = True
            else:
                if pattern_lower in rel_path_lower:
                    is_match = True

            if is_match:
                parent = str(rel_path.parent)
                if parent == ".":
                    parent = "/"
                else:
                    parent = "/" + parent

                if parent not in matches:
                    matches[parent] = []

                display_name = p.name + ("/" if p.is_dir() else "")
                matches[parent].append(display_name)

        if not matches:
            return f"No matches found for '{pattern}'."

        lines = [f"### Search results for '{pattern}':"]
        for parent in sorted(matches.keys()):
            lines.append(f"\n📁 **{parent}**")
            for item in sorted(matches[parent]):
                lines.append(f"  - {item}")

        return "\n".join(lines)
