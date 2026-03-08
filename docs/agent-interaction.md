# Agent Interaction

SemaTree exposes the **Explicit Toolset** via the Model Context Protocol (MCP). This toolset gives agents fine-grained control over how much context they consume at each step.

---

## The Explicit Toolset

Four tools are available:

### `ls(path, depth, include_summaries)`

Lists the contents of a directory node.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | string | `"/"` | Directory path to list. Must start with `/`. |
| `depth` | int | `1` | How many levels to recurse. |
| `include_summaries` | bool | `false` | If true, append the navigational snippet next to each item name. |

By default, `ls` returns **names only** — the cheapest possible listing. This is the correct first step when navigating an unknown index.

**Example output (`ls("/", depth=1)`):**
```
📁 /

### Subdirectories:
- **authentication/**
- **deployment/**
- **getting-started/**
```

**With summaries (`ls("/authentication", depth=2, include_summaries=true)`):**
```
📁 /authentication

### Subdirectories:
- **oauth/** — OAuth 2.0 flows, token exchange, and refresh handling
  - **vendor-a-oauth-guide.md**
  - **vendor-b-token-setup.md**
- **saml/** — SAML 2.0 configuration and IdP integration
```

---

### `find(pattern)`

Searches for files and directories matching a substring or glob pattern across the entire index.

| Parameter | Type | Description |
|---|---|---|
| `pattern` | string | Substring or glob pattern to match against file/directory names and paths. |

Use `find` when you already know a keyword (e.g. `find("authentication")`) to jump directly to relevant paths without drilling down level by level. Works across all sources in a multi-source index.

**Example output (`find("auth")`):**
```
### Search results for 'auth':

📁 /authentication
  - oauth/
  - saml/

📁 /authentication/oauth
  - vendor-a-oauth-guide.md
```

---

### `get_summary(path)`

Reads the YAML frontmatter and AI-generated summary for a document, or the `_summary.md` for a directory.

| Parameter | Type | Description |
|---|---|---|
| `path` | string | Path to a `.md` file or directory. Must start with `/`. |

This is the **"peek" tool** — it reads locally stored content and makes no network requests. Use it to verify a document is relevant before paying the cost of fetching the full live content.

**Example output (`get_summary("/authentication/oauth/vendor-a-oauth-guide.md")`):**
```
---
id: 0.0.1.2
title: "OAuth 2.0 Guide"
nav_summary: "OAuth flows, token exchange, and refresh token handling"
ref: https://vendor-a.example.com/docs/oauth
ref_type: url
source_id: vendor-a
content_hash: a3f1c8b2d4e7...
---

# OAuth 2.0 Guide

Covers the full OAuth 2.0 authorization code flow for Vendor A...
```

---

### `get_details(path)`

Fetches the **full content** of a document from its original source.

| Parameter | Type | Description |
|---|---|---|
| `path` | string | Path to a `.md` file (must have been returned by a prior `ls` call). |

The behaviour depends on the document's `ref_type`:

- **`ref_type: url`** — performs an HTTP GET to the original URL and returns the content as Markdown. Results are cached to `.cache/full_content/` after the first fetch.
- **`ref_type: file`** — reads the file directly from the local filesystem. No network request is made.

Use this when you need exact technical details, code examples, or step-by-step instructions that may not be fully captured in the AI summary.

---

## Usage Patterns

### Top-down drill-down

The standard pattern for exploring an unfamiliar index:

```
ls("/")                                         # See top-level categories
ls("/authentication")                           # Drill into a category
get_summary("/authentication/oauth/guide.md")   # Peek before fetching
get_details("/authentication/oauth/guide.md")   # Fetch full content if relevant
```

### Find-then-drill

When you have a specific keyword in mind:

```
find("rate limit")                              # Locate matching paths
get_summary("/api/rate-limits.md")              # Confirm relevance
get_details("/api/rate-limits.md")              # Get full content
```

### Peek-before-fetch

Use `get_summary` to scan several candidates before committing to `get_details`:

```
ls("/api", include_summaries=true)              # See snippets for all docs
get_summary("/api/streaming.md")               # Confirm it covers your topic
get_details("/api/streaming.md")               # Fetch only if confirmed
```

### Multi-source navigation

In an index with multiple sources, the structure is the same regardless of whether the index was built in source-driven or semantic mode. Use `find` or `ls` normally — the `source_id` in `get_summary` output tells you which source a document came from if you need to know:

```
find("deployment")                             # Returns results across all sources
get_summary("/deployment/vendor-a-k8s.md")    # source_id: vendor-a
get_summary("/deployment/vendor-b-docker.md") # source_id: vendor-b
```

---

## Why "Explicit" Toolset

The Explicit toolset name comes from the evaluation study. It was compared against:

- **Navigational** — `ls` injects summaries automatically, reducing tool calls but increasing context per call.
- **Simplified** — only 2 tools (`ls` and `get_details`), no `get_summary` peek step.

The Explicit toolset performed best overall because the agent retains full control over context cost at each step. The `get_summary` peek is critical: without it, the agent must either make expensive `get_details` calls speculatively or rely solely on short navigational snippets.

---

## Claude Desktop Configuration

Add the following to your Claude Desktop `claude_desktop_config.json`:

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

Replace `/absolute/path/to/my_docs` with the path to your index root directory.

You can also set the index path via environment variable instead of passing it as an argument:

```json
{
  "mcpServers": {
    "my-docs": {
      "command": "uv",
      "args": ["run", "sema-tree", "serve"],
      "env": {
        "SEMA_TREE_PATH": "/absolute/path/to/my_docs"
      }
    }
  }
}
```
