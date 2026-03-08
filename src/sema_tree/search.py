"""Keyword search over an SemaTree."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from sema_tree.models import SemaTree, IndexNode


@dataclass
class SearchResult:
    """A single search result."""

    node_id: str
    title: str
    summary: str
    ref: str | None
    score: float


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _collect_nodes(node: IndexNode) -> list[IndexNode]:
    """Flatten the tree into a list of all nodes."""
    nodes = [node]
    for child in node.children:
        nodes.extend(_collect_nodes(child))
    return nodes


def search_index(
    index: SemaTree,
    query: str,
    max_results: int = 10,
) -> list[SearchResult]:
    """Search across all node titles and summaries using TF-IDF-like scoring.

    Scoring: for each query term, score += 3 if term appears in title, 1 if in summary.
    Scores are normalized by the number of query terms so longer queries don't
    automatically get higher raw scores.
    """
    query_terms = _tokenize(query)
    if not query_terms:
        return []

    all_nodes = _collect_nodes(index.root)

    # Build document frequency for IDF-like weighting
    num_docs = len(all_nodes)
    doc_freq: dict[str, int] = {}
    node_title_tokens: dict[str, set[str]] = {}
    node_summary_tokens: dict[str, set[str]] = {}

    for node in all_nodes:
        title_tokens = set(_tokenize(node.title))
        summary_tokens = set(_tokenize(node.summary))
        node_title_tokens[node.id] = title_tokens
        node_summary_tokens[node.id] = summary_tokens
        all_tokens = title_tokens | summary_tokens
        for token in all_tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1

    results: list[SearchResult] = []
    for node in all_nodes:
        title_tokens = node_title_tokens[node.id]
        summary_tokens = node_summary_tokens[node.id]
        score = 0.0

        for term in query_terms:
            if term not in doc_freq:
                continue
            idf = math.log(num_docs / doc_freq[term]) + 1.0
            term_score = 0.0
            if term in title_tokens:
                term_score += 3.0
            if term in summary_tokens:
                term_score += 1.0
            score += term_score * idf

        if score > 0:
            # Normalize by number of query terms
            score /= len(query_terms)
            results.append(
                SearchResult(
                    node_id=node.id,
                    title=node.title,
                    summary=node.summary,
                    ref=node.ref,
                    score=round(score, 4),
                )
            )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:max_results]
