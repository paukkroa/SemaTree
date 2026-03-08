# SemaTree

SemaTree turns documentation websites and local folders into navigable semantic file trees that AI agents can explore with simple tools. Instead of retrieving chunks statistically, agents navigate a clean directory hierarchy to find exactly what they need — with precise control over how much context they consume at each step.

---

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or standard pip
pip install -e .
```

---

## LLM Provider Setup

SemaTree uses an LLM to structure and summarise the index during the build phase. Nine providers are supported via `--provider`.

### Auto-detection (default)

With `--provider auto` (the default), SemaTree tries Ollama first, then falls back to Gemini if `GEMINI_API_KEY` is set.

### Ollama (local, no API key)

Install [Ollama](https://ollama.com), pull a model, and SemaTree will find it automatically:

```bash
ollama pull llama3.2
sema-tree build https://docs.example.com my_docs  # uses Ollama automatically
```

### Gemini

```bash
export GEMINI_API_KEY=your_key_here
sema-tree build https://docs.example.com my_docs --provider gemini
```

### OpenAI

```bash
pip install 'sema-tree[providers]'
export OPENAI_API_KEY=your_key_here
sema-tree build https://docs.example.com my_docs --provider openai
```

Default model: `gpt-4o-mini`. Override with `--model gpt-4o`.

### Anthropic

```bash
pip install 'sema-tree[providers]'
export ANTHROPIC_API_KEY=your_key_here
sema-tree build https://docs.example.com my_docs --provider anthropic
```

Default model: `claude-haiku-4-5-20251001`. Override with `--model claude-sonnet-4-6`.

### OpenRouter

```bash
pip install 'sema-tree[providers]'
export OPENROUTER_API_KEY=your_key_here
sema-tree build https://docs.example.com my_docs --provider openrouter --model mistralai/mistral-7b-instruct
```

Default model: `openai/gpt-4o-mini`. Accepts any model slug from [openrouter.ai/models](https://openrouter.ai/models).

### LiteLLM

LiteLLM is a proxy layer that supports 100+ models using provider-specific env vars (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`):

```bash
pip install 'sema-tree[providers]'
sema-tree build https://docs.example.com my_docs --provider litellm --model anthropic/claude-haiku-4-5-20251001
```

Default model: `gpt-4o-mini`. Pass models in LiteLLM's `provider/model` format.

### HuggingFace Inference API

```bash
pip install 'sema-tree[providers]'
export HF_TOKEN=your_token_here
sema-tree build https://docs.example.com my_docs --provider huggingface --model mistralai/Mistral-7B-Instruct-v0.3
```

Default model: `meta-llama/Meta-Llama-3-8B-Instruct`. Also reads `HUGGINGFACE_API_KEY` as a fallback.

### llama.cpp (local server)

Start a llama.cpp server with its OpenAI-compatible endpoint, then:

```bash
sema-tree build https://docs.example.com my_docs --provider llamacpp
```

Default base URL: `http://localhost:8080`. Override with `LLAMACPP_BASE_URL` env var. No extra packages needed — uses `httpx` which is already a core dependency.

---

### Provider reference

| Provider | `--provider` | Key env var | Extra install |
|---|---|---|---|
| Auto (Ollama → Gemini) | `auto` | — | — |
| Ollama | `ollama` | — | — |
| Google Gemini | `gemini` | `GEMINI_API_KEY` | — |
| OpenAI | `openai` | `OPENAI_API_KEY` | `[providers]` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | `[providers]` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `[providers]` |
| LiteLLM | `litellm` | provider-specific | `[providers]` |
| HuggingFace | `huggingface` | `HF_TOKEN` | `[providers]` |
| llama.cpp | `llamacpp` | `LLAMACPP_BASE_URL` (opt.) | — |

---

## Quick Start

### 1. Initialize a knowledge base

```bash
sema-tree init my_docs
```

### 2. Add a documentation source

```bash
# From a website
sema-tree add https://docs.example.com/guide my_docs

# From a local folder
sema-tree add ./path/to/local/docs my_docs
```

SemaTree crawls the source (using `llms.txt` when available, BFS otherwise for web; recursive file walk for local paths), groups pages into semantic categories with an LLM, summarises each node bottom-up, and writes the result as a directory tree:

```
my_docs/
├── _meta.json
├── _summary.md
├── getting-started/
│   ├── _summary.md
│   ├── installation.md
│   └── quickstart.md
├── configuration/
│   ├── _summary.md
│   ├── environment-variables.md
│   └── provider-setup.md
└── api-reference/
    ├── _summary.md
    └── authentication.md
```

Each subsequent `add` appends a new source's subtree alongside existing ones and updates `_meta.json`.

### 3. Serve to agents

```bash
sema-tree serve my_docs
```

This starts an MCP server that exposes four tools to the agent (see [MCP Tools](#mcp-tools-explicit-toolset) below).

---

## Index Management

| Command | Description |
|---|---|
| `sema-tree init <path>` | Create a new empty index root directory |
| `sema-tree add <source> <root>` | Crawl a source and add it to an existing index root |
| `sema-tree build <source> [-o <path>]` | Build a standalone index from a single source |
| `sema-tree build-multi <s1> <s2> ... [-o <path>]` | Build an index from multiple sources (optionally with semantic grouping) |
| `sema-tree update <source-id> <index>` | Re-crawl and incrementally update a source |
| `sema-tree serve <index>` | Start the MCP server |

Sources can be a website URL (`https://...`) or a local directory path.

### Incremental updates

The `update` command only re-processes pages that actually changed since the last crawl:

```bash
sema-tree update my-source-id my_docs
```

Changed pages are detected by SHA-256 content hash. Only modified leaves and their ancestor branches are re-summarised. Pass `--restructure` to force a full rebuild instead:

```bash
sema-tree update my-source-id my_docs --restructure
```

### Semantic (cross-source) indexing

By default, each source gets its own subtree (`root → source → categories → leaves`). Use `build-multi --structure-mode semantic` to merge all sources into a single topic hierarchy regardless of source origin:

```bash
sema-tree build-multi ./docs/technical ./docs/hr ./docs/finance \
    --structure-mode semantic -o my_docs
```

In semantic mode, pages from different sources can share the same category (e.g. an "Authentication" folder may contain pages from three vendors). Each leaf retains a `source_id` field in its frontmatter so the origin is always traceable.

---

## MCP Tools (Explicit Toolset)

When served, four tools are available to the agent:

| Tool | Description | When to use |
|---|---|---|
| `ls(path, depth, include_summaries)` | List directory contents by name | First step when exploring; add `include_summaries=true` for quick orientation |
| `find(pattern)` | Search all node names/paths for a substring or glob | When you know a keyword and want to jump directly to it |
| `get_summary(path)` | Read YAML frontmatter + AI summary (local, no network) | Peek at a document before committing to a full fetch |
| `get_details(path)` | Fetch the full content — from the original URL or from disk for local sources | When you need exact technical details or code examples |

The key design: `ls` returns **names only** by default, `get_summary` is a cheap local read, and `get_details` is the only tool that may make a network request (skipped entirely for local file sources). Agents control context cost at every step.

For full usage patterns (top-down drill-down, find-then-drill, peek-before-fetch), see [docs/agent-interaction.md](docs/agent-interaction.md).

---

## Claude Desktop Configuration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "my-docs": {
      "command": "uv",
      "args": [
        "run",
        "sema-tree",
        "serve",
        "/absolute/path/to/my_docs"
      ]
    }
  }
}
```

---

## Tree Structure

Each index is a standard filesystem directory of Markdown files. Group nodes are directories with a `_summary.md`; leaf nodes are `.md` files with YAML frontmatter containing the `ref` (URL or file path), title, AI-generated summaries, source identifier, and content hash.

See [docs/tree-structure.md](docs/tree-structure.md) for the full node format and an explanation of LLM-structured, path-based, and semantic cross-source trees.

---

## Evaluation

SemaTree was evaluated against a Hybrid RAG baseline (ChromaDB + BM25/RRF) on 40 human-judged questions across a 450-document noise-rich corpus (Claude Code docs + OpenAI Codex + Python stdlib + Purina dog breeds).

**Key results:**
- Up to **~18% improvement in precision and ~19% in recall** on procedural tasks vs Hybrid RAG
- **1.00 precision** on multi-hop tasks with the Explicit toolset
- Quality **stable across corpus sizes** — hierarchy shields the agent from haystack growth
- SemaTree retrieved **1.17 sources per query** vs **3.9–4.2 for RAG** (mostly noise)

See [evaluation/README.md](evaluation/README.md) for methodology, dataset details, and how to reproduce the experiment.

---

## Development

Run tests:

```bash
uv run pytest
```

Linting:

```bash
ruff check .
```

Install dev, eval, and optional provider dependencies:

```bash
uv pip install -e ".[dev,eval]"        # dev tools + eval extras
uv pip install -e ".[providers]"       # OpenAI, Anthropic, LiteLLM, HuggingFace
uv pip install -e ".[dev,providers]"   # everything
```
