# How AgentIndex Works

AgentIndex transforms documentation sources into navigable semantic file trees through a four-stage pipeline: **Crawl → Structure → Summarize → Assemble**.

---

## Stage 1: Crawl

The crawl stage ingests raw content from either web sources or local filesystems.

### Web crawler (`crawlers/web.py`)

The `WebCrawler` uses a two-path strategy:

1. **llms.txt fast-path** — checks `{base_url}/llms.txt` first. If found, it parses the markdown link list and fetches each page in parallel (up to `--concurrency` workers). Descriptions adjacent to each link are extracted and passed downstream to help with structuring.

2. **BFS fallback** — if no `llms.txt` exists, a breadth-first crawl starts from the base URL and stays within the same domain and URL path prefix. The crawler tries an `.md` variant of each URL before falling back to HTML.

Both paths apply deduplication by final URL (after redirects) and content fingerprint, and filter out soft-404 pages and non-documentation UI pages (e.g. GitHub repo chrome).

### Local crawler (`crawlers/local.py`)

The `LocalCrawler` walks a directory recursively, reading `.md`, `.txt`, and other text files. File paths become the `ref` field for each node.

---

## Stage 2: Structure

The structure stage organises the flat list of crawled pages into a hierarchical skeleton tree.

### LLM-based structurer (`structurers/llm_based.py`)

The `LLMStructurer` is used when there are more than 7 pages. It works recursively:

1. Send the LLM a numbered list of page titles and descriptions.
2. Ask it to group them into **3–8 semantically meaningful categories** (the exact range scales with the number of pages). The LLM is explicitly forbidden from creating catch-all categories like "Other" or "Miscellaneous".
3. For any category with more than 7 pages, recursively sub-group it (up to **max depth 4**).
4. Post-process the result: merge any catch-all duplicates, dissolve sub-categories with fewer than 2 pages, and collapse single-child wrappers.

The output is a `SkeletonNode` tree: a lightweight structure that holds page titles, source URLs, and the original `CrawledPage` objects.

### Path-based structurer (`structurers/path_based.py`)

The `PathStructurer` is used as a fallback when no LLM provider is configured. It builds a deterministic trie from the URL path segments (or filesystem path components), creating one directory node per path segment. This is fast and requires no LLM calls, but the resulting hierarchy mirrors the original URL structure rather than semantic meaning.

### Auto-selection

`auto_select_structurer()` picks the LLM structurer when a provider is available and the page count warrants it, otherwise falling back to path-based.

---

## Stage 3: Summarize

The summarizer converts the `SkeletonNode` skeleton into the final `IndexNode` tree with AI-generated summaries. It works **bottom-up**: leaves are processed first, then their parents.

Each node gets two summaries:

- **DETAILED** — a comprehensive paragraph covering the technical content of the page (for leaf nodes) or the scope of the entire sub-tree (for branch nodes).
- **NAVIGATIONAL** — a 10–15 word snippet used in directory listings when `ls(include_summaries=True)` is called.

**Leaf nodes** are summarised from their raw page content (first 4000 characters). If the page came from `llms.txt` with a description, that description is reused to skip an LLM call.

**Branch nodes** are summarised from the NAVIGATIONAL snippets of their children — a "summary-of-summaries" approach that lets the LLM understand the scope of an entire sub-tree without re-reading every document.

LLM calls are made with up to 3 concurrent workers and 3 retries with exponential backoff.

---

## Stage 4: Assemble

The assembler (`fs_store.py`) walks the `IndexNode` tree and writes it to disk as a physical directory structure.

- **Branch nodes** become directories, with a `_summary.md` file containing the detailed summary and the navigational snippet as an HTML comment header.
- **Leaf nodes** become `.md` files with YAML frontmatter (see [tree-structure.md](tree-structure.md)).
- **Root/Index nodes** (id `"0"`, title `"Root"`) are collapsed — they do not create an extra directory level.
- A `_meta.json` file is written at the root with version and timestamp metadata.

The resulting directory is a standard filesystem tree that can be committed to git, shared, and served directly by the MCP server.
