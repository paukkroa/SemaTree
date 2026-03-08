"""MCP server exposing Agentic Index tools (Filesystem Backend)."""

from __future__ import annotations

import os
import sys
import re
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

from agentic_index.fs_store import FileSystemStore

MAX_CONTENT_LENGTH = 8000

mcp = FastMCP(
    "agentic-index",
    instructions=(
        "AgentIndex Explicit Toolset: use ls() to browse the index by name, "
        "get_summary() to peek at a node before committing to it, "
        "find() to search by keyword or pattern, and "
        "get_details() to fetch the full live content of a document."
    ),
)

_store: FileSystemStore | None = None


def _get_store() -> FileSystemStore:
    """Return the loaded store, raising if not initialized."""
    global _store
    if _store is None:
        path = os.environ.get("AGENTIC_INDEX_PATH")
        if not path:
            # Default to current directory if not set
            path = "."
        _store = FileSystemStore(path)
    return _store


def load_store(path: str) -> None:
    """Explicitly load a store from a root directory."""
    global _store
    _store = FileSystemStore(path)


@mcp.tool()
def ls(path: str = "/", depth: int = 1, include_summaries: bool = False) -> str:
    """List the documentation hierarchy.
    
    Args:
        path: The internal directory path to list (e.g. '/' or '/source/category'). Must start with '/'.
        depth: How many levels to list recursively (default 1).
        include_summaries: If true, includes a very concise summary for each item (default false).
        
    Returns: A list of subdirectories and document filenames.
    """
    store = _get_store()
    return store.list_dir(path, depth=depth, include_summaries=include_summaries)


@mcp.tool()
def find(pattern: str) -> str:
    """Search for files and directories matching a pattern.
    
    Args:
        pattern: A substring to search for in filenames and directory paths.
        
    Returns: A list of matches grouped by parent directory.
    """
    store = _get_store()
    return store.find(pattern)


@mcp.tool()
def get_summary(path: str) -> str:
    """Read the metadata and a brief AI-generated summary for a document.
    
    Args:
        path: The internal path to the file (e.g. '/source/doc.md'). Must start with '/'.
        
    Returns: The AI-generated summary and source metadata. 
    Use this to quickly decide if a document is relevant before fetching full details.
    """
    store = _get_store()
    return store.read_file(path)


@mcp.tool()
async def get_details(path: str) -> str:
    """Retrieve the full, live content of a documentation page.

    Args:
        path: The internal path to the document (e.g. '/source/doc.md'). Must start with '/'.
              CRITICAL: Only use a path if it was explicitly listed in a previous 'ls' call.

    Returns: The complete, latest content of the page converted to Markdown.
    ALWAYS use this when you need specific technical details, code examples, or
    step-by-step instructions.
    """
    store = _get_store()
    local_content = store.read_file(path)

    # Parse frontmatter to find 'ref' and 'ref_type'
    ref_match = re.search(r"^ref: (.+)$", local_content, re.MULTILINE)
    if not ref_match:
        return f"Error: No 'ref' found in metadata for '{path}'."

    ref = ref_match.group(1).strip()

    ref_type_match = re.search(r"^ref_type: (.+)$", local_content, re.MULTILINE)
    ref_type = ref_type_match.group(1).strip() if ref_type_match else "url"

    # Local file: read directly from filesystem
    if ref_type == "file":
        try:
            content = Path(ref).read_text(encoding="utf-8")
            if len(content) > MAX_CONTENT_LENGTH:
                content = content[:MAX_CONTENT_LENGTH] + "\n\n... [truncated]"
            return f"Local file content for '{path}' ({ref}):\n\n{content}"
        except Exception as e:
            return f"Error reading local file {ref}: {e}"

    # URL: HTTP fetch with caching
    url = ref
    cache_dir = Path(".cache/full_content")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_name = path.strip("/").replace("/", "-")
    if cache_name.endswith(".md"):
        cache_name = cache_name[:-3]
    cache_file = cache_dir / f"{cache_name}.md"

    if cache_file.exists():
        return f"Live content for '{path}' ({url}) [cached]:\n\n" + cache_file.read_text(encoding="utf-8")

    headers = {"User-Agent": "AgenticIndex/0.1"}
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0, headers=headers) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content = resp.text
    except Exception as e:
        return f"Error fetching live content from {url}: {e}"

    # Simple HTML cleanup
    if "<html" in content[:500].lower():
        try:
            from bs4 import BeautifulSoup
            from markdownify import markdownify
            soup = BeautifulSoup(content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            main = soup.find("main") or soup.find("article") or soup.body or soup
            content = markdownify(str(main), heading_style="ATX").strip()
        except ImportError:
            pass

    cache_file.write_text(content, encoding="utf-8")

    if len(content) > MAX_CONTENT_LENGTH:
        content = content[:MAX_CONTENT_LENGTH] + "\n\n... [truncated]"

    return f"Live content for '{path}' ({url}):\n\n{content}"


def main() -> None:
    """Entry point for running the MCP server."""
    # Allow passing index path as CLI argument
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        load_store(sys.argv[1])
    else:
        # Check env var or default
        _get_store()

    mcp.run()


if __name__ == "__main__":
    main()
