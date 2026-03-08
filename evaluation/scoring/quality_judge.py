"""LLM-as-judge scoring for answer quality."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from sema_tree.llm import GeminiProvider, LLMProvider, get_provider
from evaluation.config import JUDGE_MODEL


@dataclass
class JudgeScores:
    """Rubric scores from the LLM judge."""

    correctness: int  # 0-3
    completeness: int  # 0-3
    relevance: int  # 0-2
    groundedness: int  # 0-2

    @property
    def composite(self) -> float:
        """Composite score on a 0-10 scale."""
        return self.correctness + self.completeness + self.relevance + self.groundedness

    def to_dict(self) -> dict[str, float]:
        return {
            "correctness": self.correctness,
            "completeness": self.completeness,
            "relevance": self.relevance,
            "groundedness": self.groundedness,
            "composite": self.composite,
        }


JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for a documentation question-answering system.
Score the given answer against the gold-standard answer using the rubric below.

## Rubric

**Correctness (0-3):**
- 0: Factually wrong or contradicts the gold answer
- 1: Partially correct but contains significant errors
- 2: Mostly correct with minor inaccuracies
- 3: Fully correct and consistent with the gold answer

**Completeness (0-3):**
- 0: Misses all key points from the gold answer
- 1: Covers less than half the key points
- 2: Covers most key points but misses some
- 3: Covers all key points from the gold answer

**Relevance (0-2):**
- 0: Answer is off-topic or does not address the question
- 1: Partially addresses the question
- 2: Directly and fully addresses the question

**Groundedness (0-2):**
- 0: Makes claims not supported by the retrieved content
- 1: Mostly grounded but includes some unsupported claims
- 2: All claims are supported by the retrieved content

Respond with ONLY a JSON object:
{"correctness": <int>, "completeness": <int>, "relevance": <int>, "groundedness": <int>}
"""


async def judge_answer(
    question: str,
    gold_answer: str,
    candidate_answer: str,
    retrieved_content: str,
    provider: LLMProvider | None = None,
) -> JudgeScores:
    """Use an LLM as a judge to score a candidate answer against a gold answer."""
    llm = provider or get_provider()

    user_prompt = (
        f"## Question\n{question}\n\n"
        f"## Gold-standard answer\n{gold_answer}\n\n"
        f"## Retrieved content\n{retrieved_content}\n\n"
        f"## Candidate answer to evaluate\n{candidate_answer}"
    )

    resp = await llm.generate(
        user_message=user_prompt,
        system=JUDGE_SYSTEM_PROMPT,
        max_tokens=256,
    )

    raw = resp.text
    # Extract JSON from the response
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Judge did not return valid JSON: {raw}")

    scores = json.loads(match.group())
    return JudgeScores(
        correctness=int(scores["correctness"]),
        completeness=int(scores["completeness"]),
        relevance=int(scores["relevance"]),
        groundedness=int(scores["groundedness"]),
    )
