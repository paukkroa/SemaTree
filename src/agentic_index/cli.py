"""Command-line interface for Agentic Index."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click

from agentic_index.llm import PROVIDER_NAMES
from agentic_index.models import AgenticIndex


@click.group()
@click.version_option(package_name="agentic-index")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def cli(verbose: bool) -> None:
    """Agentic Index — hierarchical tree index for agentic document retrieval."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


@cli.command()
@click.argument("path", default="index_root")
def init(path: str) -> None:
    """Initialize a new multi-source index root directory."""
    root = Path(path)
    if root.exists():
        if not root.is_dir():
            click.echo(f"Error: '{path}' exists and is not a directory.")
            sys.exit(1)
        click.echo(f"Directory '{path}' already exists.")
    else:
        root.mkdir(parents=True)
        click.echo(f"Created index root at '{path}'")

    # Create root metadata
    meta_file = root / "_meta.json"
    if not meta_file.exists():
        meta = {
            "type": "agentic_index_root",
            "version": "1.0",
            "sources": []
        }
        meta_file.write_text(json.dumps(meta, indent=2))
    
    # Create root summary
    summary_file = root / "_summary.md"
    if not summary_file.exists():
        summary_file.write_text("# Knowledge Base Root\n\nThis index contains documentation from multiple sources.\n")


@cli.command()
@click.argument("source")
@click.option("-o", "--output", default="index_root", help="Output path (directory for file tree, .json for legacy file).")
@click.option(
    "--provider", default="auto", type=click.Choice(PROVIDER_NAMES),
    help="LLM provider for summarization (default: auto-detect).",
)
@click.option("--model", default=None, help="Model name override for the LLM provider.")
@click.option("--max-pages", default=200, help="Maximum number of pages to crawl.")
@click.option("--concurrency", default=10, help="Number of concurrent crawl requests.")
def build(source: str, output: str, provider: str, model: str | None, max_pages: int, concurrency: int) -> None:
    """Build a new index from a source URL or local path."""
    from agentic_index.builder import IndexBuilder
    from agentic_index.llm import get_provider
    from agentic_index.fs_store import FileSystemStore

    async def _run() -> None:
        llm = get_provider(provider=provider, model=model)
        click.echo(f"Using LLM provider: {llm}")
        builder = IndexBuilder(provider=llm)
        index = await builder.build(source, max_pages=max_pages, concurrency=concurrency)
        
        if output.endswith(".json"):
            # Legacy single-file mode
            index.save(output)
            click.echo(f"Index saved to {output} (JSON format)")
        else:
            # New filesystem mode
            store = FileSystemStore(output)
            store.save(index)
            click.echo(f"Index saved to directory: {output} (Filesystem format)")
            
        _print_stats(index)

    asyncio.run(_run())


@cli.command()
@click.argument("source")
@click.argument("root_path", metavar="ROOT_PATH")
@click.option(
    "--provider", default="auto", type=click.Choice(PROVIDER_NAMES),
    help="LLM provider for summarization.",
)
@click.option("--model", default=None, help="Model name override.")
@click.option("--max-pages", default=200, help="Maximum number of pages to crawl.")
@click.option("--concurrency", default=10, help="Number of concurrent crawl requests.")
def add(source: str, root_path: str, provider: str, model: str | None, max_pages: int, concurrency: int) -> None:
    """Add a new source to an existing index root directory."""
    from agentic_index.builder import IndexBuilder
    from agentic_index.composer import add_source
    from agentic_index.fs_store import FileSystemStore
    from agentic_index.llm import get_provider
    from agentic_index.models import AgenticIndex

    root = Path(root_path)
    if not root.exists() or not root.is_dir():
        click.echo(f"Error: Root path '{root_path}' does not exist or is not a directory. Run 'init' first.")
        sys.exit(1)

    async def _run() -> None:
        llm = get_provider(provider=provider, model=model)
        click.echo(f"Using LLM provider: {llm}")

        store = FileSystemStore(root)

        # Load existing index (or create empty one for a freshly init'd root)
        meta_file = root / "_meta.json"
        if meta_file.exists():
            try:
                index = store.load(root)
            except Exception as e:
                click.echo(f"Warning: could not load existing index ({e}). Starting fresh.")
                index = AgenticIndex()
        else:
            index = AgenticIndex()

        # Add the new source to the in-memory index
        builder = IndexBuilder(provider=llm)
        index = await add_source(index, source, builder=builder)

        # Save the updated index (writes new source files + updates _meta.json)
        store.save(index)

        click.echo(f"Source '{source}' added to {root}")
        _print_stats(index)

    asyncio.run(_run())


@cli.command()
@click.argument("source_id")
@click.argument("index_path", metavar="INDEX")
@click.option(
    "--provider", default="auto", type=click.Choice(PROVIDER_NAMES),
    help="LLM provider for summarization (default: auto-detect).",
)
@click.option("--model", default=None, help="Model name override for the LLM provider.")
@click.option(
    "--restructure", is_flag=True, default=False,
    help="Force a full rebuild instead of an incremental update.",
)
def update(source_id: str, index_path: str, provider: str, model: str | None, restructure: bool) -> None:
    """Re-crawl and update an existing source in the index."""
    from agentic_index.builder import IndexBuilder
    from agentic_index.composer import update_source
    from agentic_index.fs_store import FileSystemStore
    from agentic_index.llm import get_provider

    async def _run() -> None:
        llm = get_provider(provider=provider, model=model)
        click.echo(f"Using LLM provider: {llm}")
        builder = IndexBuilder(provider=llm)

        if Path(index_path).is_dir():
            store = FileSystemStore(index_path)
            index = store.load(index_path)
            index = await update_source(
                index, source_id, builder=builder, incremental=not restructure
            )
            store.save(index)
        else:
            index = AgenticIndex.load(index_path)
            index = await update_source(
                index, source_id, builder=builder, incremental=not restructure
            )
            index.save(index_path)

        click.echo(f"Source '{source_id}' updated. Index saved to {index_path}")
        _print_stats(index)

    asyncio.run(_run())


@cli.command("build-multi")
@click.argument("sources", nargs=-1, required=True)
@click.option("-o", "--output", default="index_root", help="Output path (directory).")
@click.option(
    "--structure-mode", default="source",
    type=click.Choice(["source", "semantic"]),
    help="'source' (default) keeps each source as its own subtree. "
         "'semantic' merges all pages into one cross-source hierarchy.",
)
@click.option(
    "--provider", default="auto", type=click.Choice(PROVIDER_NAMES),
    help="LLM provider for summarization (default: auto-detect).",
)
@click.option("--model", default=None, help="Model name override for the LLM provider.")
@click.option("--max-pages", default=200, help="Maximum pages per source.")
@click.option("--concurrency", default=10, help="Concurrent requests per source (web only).")
def build_multi(
    sources: tuple[str, ...],
    output: str,
    structure_mode: str,
    provider: str,
    model: str | None,
    max_pages: int,
    concurrency: int,
) -> None:
    """Build an index from multiple sources with optional semantic grouping.

    SOURCES can be URLs or local paths. Example:

    \b
        agentic-index build-multi ./docs/technical ./docs/hr \\
            --structure-mode semantic -o /tmp/combined_idx
    """
    from agentic_index.builder import IndexBuilder
    from agentic_index.fs_store import FileSystemStore
    from agentic_index.llm import get_provider

    async def _run() -> None:
        llm = get_provider(provider=provider, model=model)
        click.echo(f"Using LLM provider: {llm}")
        builder = IndexBuilder(provider=llm)
        index = await builder.build_multi(
            list(sources),
            structure_mode=structure_mode,
            max_pages=max_pages,
            concurrency=concurrency,
        )

        store = FileSystemStore(output)
        store.save(index)
        click.echo(f"Index saved to directory: {output}")
        _print_stats(index)

    asyncio.run(_run())


@cli.command()
@click.argument("index_path", metavar="INDEX")
@click.option("--node", default="0", help="Node ID to explore (default: root).")
def explore(index_path: str, node: str) -> None:
    """Interactively explore the index tree."""
    if Path(index_path).is_dir():
        click.echo("Interactive exploration not yet supported for directory indexes. Use 'serve' or standard 'ls' commands.")
        return

    index = AgenticIndex.load(index_path)
    current_id = node

    while True:
        current = index.root.find_node(current_id)
        if current is None:
            click.echo(f"Node '{current_id}' not found.")
            break

        click.echo(f"\n[{current.id}] {current.title}")
        if current.summary:
            click.echo(f"  {current.summary}")
        if current.ref:
            click.echo(f"  ref: {current.ref}")

        if current.children:
            click.echo(f"\nChildren ({len(current.children)}):")
            for i, child in enumerate(current.children):
                click.echo(f"  {i}: [{child.id}] {child.title}")
                if child.summary:
                    click.echo(f"     {child.summary}")

        click.echo()
        click.echo("Enter child number, node ID, 'u' for parent, or 'q' to quit:")
        try:
            choice = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "q":
            break
        elif choice == "u":
            parts = current_id.rsplit(".", 1)
            if len(parts) > 1:
                current_id = parts[0]
            else:
                click.echo("Already at root.")
        elif choice.isdigit() and current.children:
            idx = int(choice)
            if 0 <= idx < len(current.children):
                current_id = current.children[idx].id
            else:
                click.echo("Invalid child number.")
        elif "." in choice or choice == "0":
            current_id = choice
        else:
            click.echo("Invalid input.")


@cli.command("search")
@click.argument("query")
@click.argument("index_path", metavar="INDEX")
@click.option("--max-results", default=10, type=int, help="Maximum number of results.")
def search_cmd(query: str, index_path: str, max_results: int) -> None:
    """Search the index for matching nodes."""
    if Path(index_path).is_dir():
        click.echo("Search not yet supported for directory indexes.")
        return

    from agentic_index.search import search_index

    index = AgenticIndex.load(index_path)
    results = search_index(index, query, max_results=max_results)

    if not results:
        click.echo(f"No results found for '{query}'.")
        return

    click.echo(f"Results for '{query}' ({len(results)} matches):\n")
    for r in results:
        click.echo(f"  [{r.node_id}] {r.title} (score: {r.score})")
        if r.summary:
            click.echo(f"    {r.summary}")
        if r.ref:
            click.echo(f"    ref: {r.ref}")
        click.echo()


@cli.command()
@click.argument("index_path", metavar="INDEX")
@click.option(
    "--provider", default="auto", type=click.Choice(PROVIDER_NAMES),
    help="LLM provider for the chat agent.",
)
@click.option("--model", default=None, help="Model name override.")
def chat(index_path: str, provider: str, model: str | None) -> None:
    """Start an interactive chat session with the Agentic Index."""
    from agentic_index.client import AgenticChat
    from agentic_index.llm import get_provider
    from mcp import StdioServerParameters
    import os

    server_params = StdioServerParameters(
        command="uv",
        args=["run", "agentic-index", "serve", os.path.abspath(index_path)],
        env=os.environ.copy()
    )

    llm = get_provider(provider=provider, model=model)
    chat_session = AgenticChat(server_params, llm)
    
    async def _run() -> None:
        await chat_session.start()

    asyncio.run(_run())


@cli.command()
@click.argument("index_path", metavar="INDEX")
@click.option("--host", default="127.0.0.1", help="Host to bind to.")
@click.option("--port", default=8420, type=int, help="Port for the web UI.")
def ui(index_path: str, host: str, port: int) -> None:
    """Launch the web UI for browsing and managing the index."""
    if Path(index_path).is_dir():
        click.echo("Web UI not yet supported for directory indexes.")
        return

    from agentic_index.web.app import start_server

    click.echo(f"Starting web UI at http://{host}:{port}")
    start_server(index_path, host=host, port=port)


@cli.command()
@click.argument("index_path", metavar="INDEX")
def serve(index_path: str) -> None:
    """Start the MCP server for the index."""
    import os

    os.environ["AGENTIC_INDEX_PATH"] = index_path
    click.echo(f"Starting MCP server with index: {index_path}", err=True)

    from agentic_index.server.mcp_server import load_store, mcp

    # Check if it's a directory or legacy JSON file
    if Path(index_path).is_file() and index_path.endswith(".json"):
        click.echo("Error: This version of MCP server requires a directory-based index (FilesystemStore).")
        click.echo("Please rebuild your index without the .json extension.")
        sys.exit(1)
        
    load_store(index_path)
    mcp.run()


@cli.command()
@click.argument("index_path", metavar="INDEX")
def stats(index_path: str) -> None:
    """Show statistics about the index."""
    if Path(index_path).is_dir():
        click.echo("Stats not available for directory indexes.")
        return

    index = AgenticIndex.load(index_path)
    _print_stats(index)


def _print_stats(index: AgenticIndex) -> None:
    """Print index statistics."""
    root = index.root
    click.echo(f"Sources:     {len(index.sources)}")
    click.echo(f"Total nodes: {root.count_nodes()}")
    click.echo(f"Leaf nodes:  {root.count_leaves()}")
    click.echo(f"Max depth:   {root.depth}")
    click.echo(f"Created:     {index.created_at.isoformat()}")
    click.echo(f"Updated:     {index.updated_at.isoformat()}")

    if index.sources:
        click.echo("\nSources:")
        for source in index.sources:
            click.echo(f"  {source.id} ({source.type.value}): {source.origin}")
            click.echo(f"    Pages: {source.page_count}, Crawled: {source.crawled_at.isoformat()}")
