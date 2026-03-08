"""Semi-automated question generation from documentation pages."""

from __future__ import annotations

import json
import re
from pathlib import Path

from sema_tree.llm import LLMProvider, get_provider
from evaluation.config import GENERATION_MODEL, QUESTION_CATEGORIES

GENERATION_SYSTEM_PROMPT = """\
You are an expert at creating evaluation questions for documentation QA systems.
Given a documentation page, generate evaluation questions in the requested category.

## Question categories

- **SF** (Single-Fact): Questions answerable from a single sentence/paragraph in one page.
- **MH** (Multi-Hop): Questions requiring synthesis from 2+ sections or pages.
- **COMP** (Comparison): Questions asking to compare features, approaches, or options.
- **PROC** (Procedural): Questions about step-by-step processes or how to accomplish a task.

## Output format

Return a JSON array of question objects:
[
  {
    "id": "SF-001",
    "category": "SF",
    "question": "...",
    "gold_answer": "...",
    "gold_sources": ["page_slug_1"],
    "difficulty": "easy|medium|hard"
  }
]

Generate exactly the number of questions requested. Make questions specific, unambiguous, and
answerable from the provided documentation content.
"""


async def generate_questions(
    page_contents: list[dict[str, str]],
    category: str,
    count: int,
    start_id: int = 1,
    provider: LLMProvider | None = None,
) -> list[dict]:
    """Generate evaluation questions for a given category from doc pages.

    ``page_contents`` is a list of ``{"slug": ..., "title": ..., "content": ...}`` dicts.
    """
    llm = provider or get_provider()

    # Combine pages into context
    context_parts = []
    for page in page_contents:
        context_parts.append(f"## Page: {page['title']} (slug: {page['slug']})\n\n{page['content']}")
    context = "\n\n---\n\n".join(context_parts)

    user_prompt = (
        f"Generate {count} evaluation questions in the **{category}** category.\n"
        f"Use IDs starting from {category}-{start_id:03d}.\n\n"
        f"Documentation content:\n\n{context}"
    )

    resp = await llm.generate(
        user_message=user_prompt,
        system=GENERATION_SYSTEM_PROMPT,
        max_tokens=4096,
    )

    raw = resp.text
    # Extract JSON array
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Generator did not return a JSON array: {raw[:200]}")

    return json.loads(match.group())


def save_questions(questions: list[dict], output_path: Path) -> None:
    """Save generated questions to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(questions, indent=2), encoding="utf-8")


def load_questions(path: Path) -> list[dict]:
    """Load questions from a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))
