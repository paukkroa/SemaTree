# How SemaTree Works

SemaTree transforms documentation sources into navigable semantic file trees through a four-stage pipeline: **Crawl → Structure → Summarize → Assemble**. It also supports loading existing indexes back into memory, incremental updates, and cross-source semantic structuring.

---

## Stage 1: Crawl

The crawl stage ingests raw content from either web sources or local filesystems.

### Web crawler (`crawlers/web.py`)

The `WebCrawler` uses a two-path strategy:

1. **llms.txt fast-path** — checks `{base_url}/llms.txt` first. If found, it parses the markdown link list and fetches each page in parallel (up to `--concurrency` workers). Descriptions adjacent to each link are extracted and passed downstream to help with structuring.

2. **BFS fallback** — if no `llms.txt` exists, a breadth-first crawl starts from the base URL and stays within the same domain and URL path prefix. The crawler tries an `.md` variant of each URL before falling back to HTML.

Both paths apply deduplication by final URL (after redirects) and content fingerprint, and filter out soft-404 pages and non-documentation UI pages (e.g. GitHub repo chrome).

### Local crawler (`crawlers/local.py`)

The `LocalCrawler` walks a directory recursively, reading `.md`, `.txt`, `.rst`, and `.html` files. It respects `.gitignore` patterns and skips common non-content directories (`.git`, `node_modules`, `__pycache__`, etc.). File paths become the `ref` field for each node with `ref_type: file`.

---

## Stage 2: Structure

The structure stage organises the flat list of crawled pages into a hierarchical skeleton tree.

### LLM-based structurer (`structurers/llm_based.py`)

The `LLMStructurer` is used when there are more than 7 pages. It works recursively:

1. Send the LLM a numbered list of page titles and descriptions.
2. Ask it to group them into **3–8 semantically meaningful categories** (the exact range scales with the number of pages). The LLM is explicitly forbidden from creating catch-all categories like "Other" or "Miscellaneous".
3. For any category with more than 7 pages, recursively sub-group it (up to **max depth 4**).
4. Post-process the result: merge any catch-all duplicates, dissolve sub-categories with fewer than 2 pages, and collapse single-child wrappers.

The output is a `SkeletonNode` tree: a lightweight structure that holds page titles, source URLs/paths, and the original `CrawledPage` objects.

### Path-based structurer (`structurers/path_based.py`)

The `PathStructurer` is used as a fallback when no LLM provider is configured. It builds a deterministic trie from the URL path segments (or filesystem path components), creating one directory node per path segment. This is fast and requires no LLM calls, but the resulting hierarchy mirrors the original URL structure rather than semantic meaning.

### Cross-source semantic structurer (`structurers/semantic.py`)

The `CrossSourceStructurer` is used when `build-multi --structure-mode semantic` is invoked. Instead of keeping each source in its own subtree, it:

1. Flattens pages from **all sources** into a single pool.
2. Tags each page title with a `[source_id]` prefix so the LLM can disambiguate pages with similar names from different sources.
3. Runs `LLMStructurer` on the combined pool to produce one unified semantic hierarchy.
4. Restores original page titles and preserves the `original_source_id` on each leaf for downstream source attribution.

The result is a single tree where, for example, an "Authentication" category may contain pages from multiple vendors. The `source_id` field on each leaf node records which source it came from.

### Auto-selection

`auto_select_structurer()` picks the LLM structurer when a provider is available and the page count warrants it, otherwise falling back to path-based.

### LLM providers

Any of the nine supported providers can be used for the structuring and summarisation stages. The provider is selected via `--provider` on the CLI or passed directly to `IndexBuilder`. See the [README](../README.md#llm-provider-setup) for the full list and configuration details.

---

## Stage 3: Summarize

The summarizer (`summarizer.py`) converts the `SkeletonNode` skeleton into the final `IndexNode` tree with AI-generated summaries. It works **bottom-up**: leaves are processed first, then their parents.

Each node gets two summaries:

- **DETAILED** — a comprehensive paragraph covering the technical content of the page (for leaf nodes) or the scope of the entire sub-tree (for branch nodes).
- **NAVIGATIONAL** — a 10–15 word snippet used in directory listings when `ls(include_summaries=True)` is called.

**Leaf nodes** are summarised from their raw page content (first 4000 characters). If the page came from `llms.txt` with a description, that description is reused to skip an LLM call. The SHA-256 hash of the raw content is stored as `content_hash` for later change detection.

**Branch nodes** are summarised from the NAVIGATIONAL snippets of their children — a "summary-of-summaries" approach that lets the LLM understand the scope of an entire sub-tree without re-reading every document.

**Cross-source `source_id` assignment**: when pages carry an `original_source_id` in their metadata (set by `CrossSourceStructurer`), the summarizer uses that value instead of the global source ID, so each leaf node in a semantic index correctly identifies its origin source.

LLM calls are made with up to 3 concurrent workers and 3 retries with exponential backoff.

---

## Stage 4: Assemble

The assembler (`fs_store.py`) walks the `IndexNode` tree and writes it to disk as a physical directory structure.

- **Branch nodes** become directories with a `_summary.md` file. The file contains:
  - An HTML comment with the navigational snippet: `<!-- nav_summary: ... -->`
  - A metadata comment with the node ID, original title, and source ID: `<!-- meta: {"id": "0.0", "title": "Getting Started", "source_id": "my-docs"} -->`
  - The detailed summary as the body.
- **Leaf nodes** become `.md` files with YAML frontmatter (see [tree-structure.md](tree-structure.md)).
- **Root nodes** (id `"0"`, title `"Root"`) are collapsed — they do not create an extra directory level.
- A `_meta.json` file is written at the root with version, timestamps, and the full `sources` list.

The resulting directory is a standard filesystem tree that can be committed to git, shared, and served directly by the MCP server.

---

## Loading an Existing Index

`FileSystemStore.load(path)` reconstructs a full `SemaTree` from a saved directory:

1. Reads `_meta.json` for version, timestamps, and the `sources` list.
2. Walks the directory tree. For each subdirectory, reads `_summary.md` to extract the `nav_summary`, `id`, `title`, and `source_id` from the embedded comments.
3. For each `.md` file, parses the YAML frontmatter to reconstruct all `IndexNode` fields including `ref`, `ref_type`, `source_id`, and `content_hash`.
4. Sorts children by their `id` field to restore the original tree order.
5. Returns a fully populated `SemaTree`.

This is used by the `add` and `update` CLI commands to modify an existing index without starting from scratch.

---

## Incremental Updates

The `update_source()` function in `composer.py` supports an `incremental=True` mode (the default). Rather than doing a full re-crawl and re-structure, it:

1. Re-crawls the source to get the current set of pages.
2. Calls `IncrementalUpdater.compute_diff()` to classify pages as **unchanged** (same SHA-256 hash), **changed** (different hash), **added** (new ref), or **deleted** (ref absent from new crawl).
3. Calls `IncrementalUpdater.apply_diff()` to:
   - Re-summarise only the changed leaves.
   - Remove deleted leaf nodes (and clean up empty parent branches).
   - Append new leaf nodes under the source's root branch.
   - Re-summarise dirty branch nodes bottom-up.

Pages whose `content_hash` is unchanged are skipped entirely — no LLM calls, no file writes. This makes routine updates of large indexes fast and cheap.

Pass `incremental=False` (or `--restructure` on the CLI) to force a full rebuild.
