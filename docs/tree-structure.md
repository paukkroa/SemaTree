# Tree Structure

AgentIndex produces two types of directory trees depending on how the index was structured.

---

## Tree Types

### LLM-Structured Tree

Built by the `LLMStructurer`. Pages are grouped by **semantic meaning**, independent of the original URL structure. A documentation site with URLs like `/docs/v2/ref/h-723x` might produce a tree like:

```
my-docs/
├── _meta.json
├── _summary.md
└── source-name/
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
        ├── authentication.md
        └── endpoints.md
```

The folder names are chosen by the LLM to reflect the **topic** of their contents, not the URL path.

### Path-Based Tree

Built by the `PathStructurer` (fallback, no LLM required). The hierarchy mirrors the original URL path segments:

```
my-docs/
└── source-name/
    └── docs/
        └── en/
            ├── getting-started.md
            └── api/
                └── endpoints.md
```

---

## Node Anatomy

### Group Nodes (directories)

Each directory represents a conceptual category. It contains:

- `_summary.md` — the AI-generated summary for the category. The first line is an HTML comment with the navigational snippet:
  ```
  <!-- nav_summary: Install, configure, and run your first index -->

  Getting Started covers the full onboarding workflow...
  ```
- Child directories and `.md` leaf files.

### Leaf Nodes (`.md` files)

Each document is a Markdown file with YAML frontmatter:

```yaml
---
id: 0.0.1.2
title: "Environment Variables"
nav_summary: "All supported env vars and their defaults"
ref: https://docs.example.com/config/env-vars
ref_type: url
---

# Environment Variables

<detailed AI-generated summary of the page content>

[Link to original](https://docs.example.com/config/env-vars)
```

**Frontmatter fields:**

| Field | Description |
|---|---|
| `id` | Hierarchical node identifier (dot-separated) |
| `title` | Page title extracted during crawl |
| `nav_summary` | Short navigational snippet (10–15 words) |
| `ref` | Original URL or filesystem path of the source document |
| `ref_type` | `url` for web sources, `path` for local files |

The `ref` field is what `get_details()` uses to fetch live content.

---

## Root Metadata

`_meta.json` at the index root:

```json
{
  "version": "1.0",
  "created_at": "2025-01-15T10:30:00+00:00",
  "updated_at": "2025-01-15T10:30:00+00:00"
}
```
