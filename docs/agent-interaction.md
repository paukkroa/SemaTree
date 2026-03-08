# Agent Interaction

AgentIndex exposes the **Explicit Toolset** via the Model Context Protocol (MCP). This toolset gives agents fine-grained control over how much context they consume at each step.

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
- **claude-docs/**
- **python-tutorial/**
```

**With summaries (`ls("/claude-docs", depth=2, include_summaries=true)`):**
```
📁 /claude-docs

### Subdirectories:
- **getting-started/** — Install Claude Code and run your first command
  - **installation.md**
  - **quickstart.md**
- **configuration/** — Environment variables, settings, and provider setup
```

---

### `find(pattern)`

Searches for files and directories matching a substring or glob pattern across the entire index.

| Parameter | Type | Description |
|---|---|---|
| `pattern` | string | Substring or glob pattern to match against file/directory names and paths. |

Use `find` when you already know a keyword (e.g. `find("authentication")`) to jump directly to relevant paths without drilling down level by level.

**Example output (`find("auth")`):**
```
### Search results for 'auth':

📁 /claude-docs/security
  - authentication.md
  - authorization.md
```

---

### `get_summary(path)`

Reads the YAML frontmatter and AI-generated summary for a document, or the `_summary.md` for a directory.

| Parameter | Type | Description |
|---|---|---|
| `path` | string | Path to a `.md` file or directory. Must start with `/`. |

This is the **"peek" tool** — it reads locally stored content and makes no network requests. Use it to verify a document is relevant before paying the cost of fetching the full live content.

**Example output (`get_summary("/claude-docs/configuration/environment-variables.md")`):**
```
---
id: 0.0.2.1
title: "Environment Variables"
nav_summary: "All supported env vars and their defaults"
ref: https://docs.example.com/config/env-vars
ref_type: url
---

# Environment Variables

Covers CLAUDE_API_KEY, CLAUDE_MODEL, MAX_TOKENS, and all other
supported environment variables. Includes default values and examples
for each provider (Ollama, Gemini).
```

---

### `get_details(path)`

Fetches the **full, live content** of a document from its original source URL.

| Parameter | Type | Description |
|---|---|---|
| `path` | string | Path to a `.md` file (must have been returned by a prior `ls` call). |

Results are cached to `.cache/full_content/` after the first fetch. Use this when you need exact technical details, code examples, or step-by-step instructions that may not be fully captured in the AI summary.

---

## Usage Patterns

### Top-down drill-down

The standard pattern for exploring an unfamiliar index:

```
ls("/")                                    # See top-level categories
ls("/claude-docs")                         # Drill into a category
get_summary("/claude-docs/api/auth.md")    # Peek before fetching
get_details("/claude-docs/api/auth.md")    # Fetch full content if relevant
```

### Find-then-drill

When you have a specific keyword in mind:

```
find("rate limit")                         # Locate matching paths
get_summary("/claude-docs/api/rate-limits.md")  # Confirm relevance
get_details("/claude-docs/api/rate-limits.md")  # Get full content
```

### Peek-before-fetch

Use `get_summary` to scan several candidates before committing to `get_details`:

```
ls("/claude-docs/api", include_summaries=true)  # See snippets for all docs
get_summary("/claude-docs/api/streaming.md")    # Confirm it covers your topic
get_details("/claude-docs/api/streaming.md")    # Fetch only if confirmed
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
        "agentic-index",
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
      "args": ["run", "agentic-index", "serve"],
      "env": {
        "AGENTIC_INDEX_PATH": "/absolute/path/to/my_docs"
      }
    }
  }
}
```
