# Tree Structure

SemaTree produces directory trees of Markdown files. The tree shape depends on how the index was built.

---

## Tree Types

### Source-driven tree (default)

Each source occupies its own subtree under the index root. This is the default for `build`, `add`, and `build-multi --structure-mode source`:

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

After adding a second source, its subtree appears alongside the first:

```
my_docs/
├── _meta.json
├── getting-started/          ← from source 1
├── api-reference/            ← from source 1
├── hr-policies/              ← from source 2
└── benefits/                 ← from source 2
```

The folder names are chosen by the LLM to reflect the **topic** of their contents, not the URL path or source name.

### Path-based tree

Built by the `PathStructurer` (fallback, no LLM required). The hierarchy mirrors the original URL path segments:

```
my_docs/
└── docs/
    └── en/
        ├── getting-started.md
        └── api/
            └── endpoints.md
```

### Semantic cross-source tree

Built by `build-multi --structure-mode semantic`. All pages from all sources are merged into one unified topic hierarchy. Pages from different sources can share the same category:

```
my_docs/
├── _meta.json
├── authentication/
│   ├── _summary.md
│   ├── vendor-a-oauth-guide.md     ← source_id: vendor-a
│   └── vendor-b-saml-setup.md      ← source_id: vendor-b
├── deployment/
│   ├── _summary.md
│   ├── vendor-a-kubernetes.md      ← source_id: vendor-a
│   └── vendor-b-docker.md          ← source_id: vendor-b
└── getting-started/
    ├── _summary.md
    └── quickstart.md               ← source_id: vendor-a
```

The `source_id` frontmatter field on each leaf identifies which original source it came from.

---

## Node Anatomy

### Group Nodes (directories)

Each directory represents a conceptual category. It always contains a `_summary.md`:

```
<!-- nav_summary: Install, configure, and run your first index -->
<!-- meta: {"id": "0.0", "title": "Getting Started", "source_id": "my-docs"} -->

Getting Started covers the full onboarding workflow from installation through
running your first index build. Includes environment setup, provider
configuration, and your first `sema-tree build` command.
```

- The first `<!-- nav_summary: ... -->` comment is the short navigational snippet shown by `ls(include_summaries=true)`.
- The `<!-- meta: {...} -->` comment stores the node's hierarchical `id`, original `title`, and `source_id`. This is what `FileSystemStore.load()` reads to reconstruct the in-memory tree.
- The body is the detailed AI-generated summary for the category.

### Leaf Nodes (`.md` files)

Each document is a Markdown file with YAML frontmatter:

```yaml
---
id: 0.0.1.2
title: "Environment Variables"
nav_summary: "All supported env vars and their defaults"
ref: https://docs.example.com/config/env-vars
ref_type: url
source_id: my-docs
content_hash: a3f1c8b2d4e7...
---

# Environment Variables

<detailed AI-generated summary of the page content>

[Link to original](https://docs.example.com/config/env-vars)
```

**Frontmatter fields:**

| Field | Description |
|---|---|
| `id` | Hierarchical node identifier (dot-separated, e.g. `0.0.1.2`) |
| `title` | Page title extracted during crawl |
| `nav_summary` | Short navigational snippet (10–15 words) |
| `ref` | Original URL or filesystem path of the source document |
| `ref_type` | `url` for web sources, `file` for local filesystem sources |
| `source_id` | ID of the source this page belongs to (matches an entry in `_meta.json`) |
| `content_hash` | SHA-256 of the raw page content at index time; used for incremental change detection |

The `ref` field is what `get_details()` uses to retrieve the full content. For `ref_type: url` it performs an HTTP fetch; for `ref_type: file` it reads the file directly from the local filesystem.

---

## Root Metadata

`_meta.json` at the index root stores version information, timestamps, and the complete list of indexed sources:

```json
{
  "version": "1.0",
  "created_at": "2025-01-15T10:30:00+00:00",
  "updated_at": "2025-01-15T10:30:00+00:00",
  "sources": [
    {
      "id": "my-docs",
      "type": "website",
      "origin": "https://docs.example.com",
      "crawled_at": "2025-01-15T10:30:00+00:00",
      "page_count": 42
    },
    {
      "id": "local-guides",
      "type": "local_folder",
      "origin": "/home/user/guides",
      "crawled_at": "2025-01-15T11:00:00+00:00",
      "page_count": 8
    }
  ]
}
```

**Source fields:**

| Field | Description |
|---|---|
| `id` | Slug derived from the source URL or path (used as `source_id` on leaf nodes) |
| `type` | `website` or `local_folder` |
| `origin` | The original URL or filesystem path that was crawled |
| `crawled_at` | ISO 8601 timestamp of the last crawl |
| `page_count` | Number of pages indexed from this source |

`FileSystemStore.load()` reads this file to reconstruct the `SemaTree.sources` list, enabling `update` and `add` to work correctly without rebuilding from scratch.
