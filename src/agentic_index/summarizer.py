"""Bottom-up tree summarization using LLM."""

from __future__ import annotations

import asyncio
import logging
import re

from agentic_index.llm import LLMProvider, LLMResponse, get_provider
from agentic_index.models import IndexNode, RefType
from agentic_index.structurers.base import SkeletonNode

logger = logging.getLogger(__name__)

MAX_CONTENT_CHARS = 4000
LEAF_SYSTEM_PROMPT = (
    "You are a technical documentation expert. Write two summaries for the given page:\n"
    "1. DETAILED: A comprehensive paragraph covering key technical concepts and features.\n"
    "2. NAVIGATIONAL: A very concise (max 15 words) snippet for directory listings.\n\n"
    "Format your response exactly as:\n"
    "DETAILED: <detailed summary>\n"
    "NAVIGATIONAL: <navigational snippet>"
)
BRANCH_SYSTEM_PROMPT = (
    "You are a technical documentation expert. Given the child section summaries, write two summaries for this section:\n"
    "1. DETAILED: A comprehensive paragraph synthesizing the scope of the entire section.\n"
    "2. NAVIGATIONAL: A very concise (max 15 words) snippet for directory listings.\n\n"
    "Format your response exactly as:\n"
    "DETAILED: <detailed summary>\n"
    "NAVIGATIONAL: <navigational snippet>"
)


class Summarizer:
    """Bottom-up tree summarizer using LLM."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        concurrency: int = 3,
    ):
        self._provider = provider
        self._semaphore = asyncio.Semaphore(concurrency)

    def _get_provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = get_provider()
        return self._provider

    def _parse_summaries(self, text: str) -> tuple[str, str]:
        """Parse the dual-summary response format."""
        detailed = ""
        nav = ""
        
        # Flexibly find labels with optional markdown (e.g. **DETAILED:**)
        det_pattern = r"(?:\*\*)?DETAILED:(?:\*\*)?\s*(.*?)(?=(?:\*\*)?NAVIGATIONAL:(?:\*\*)?|$)"
        nav_pattern = r"(?:\*\*)?NAVIGATIONAL:(?:\*\*)?\s*(.*)"
        
        det_match = re.search(det_pattern, text, re.DOTALL | re.IGNORECASE)
        nav_match = re.search(nav_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if det_match:
            detailed = det_match.group(1).strip()
        if nav_match:
            nav = nav_match.group(1).strip()
            # Clean up if detailed followed nav in text
            nav = re.split(det_pattern, nav, flags=re.IGNORECASE)[0].strip()
            
        # Fallback if parsing fails
        if not detailed:
            detailed = text.strip()
            # Clean off any labels that might have been there
            detailed = re.sub(r"^(?:\*\*)?DETAILED:(?:\*\*)?\s*", "", detailed, flags=re.IGNORECASE)
            detailed = re.split(r"(?:\*\*)?NAVIGATIONAL:(?:\*\*)?", detailed, flags=re.IGNORECASE)[0].strip()
            
        if not nav:
            nav = detailed.split(".")[0][:150]
            
        # AGGRESSIVE CLEANING for nav_summary
        # 1. Collapse all whitespace/newlines
        nav = re.sub(r"\s+", " ", nav).strip()
        # 2. Strip leading/trailing markdown bold/italic/quote markers
        nav = re.sub(r"^[\*\"'_-]+|[\*\"'_-]+$", "", nav).strip()
        # 3. Final trim
        nav = nav[:150]
            
        return detailed, nav

    async def summarize_tree(
        self, skeleton: SkeletonNode, source_id: str, id_prefix: str = "0.0"
    ) -> IndexNode:
        """Convert a SkeletonNode tree into an IndexNode tree with summaries.

        Processes bottom-up: leaves first, then internal nodes.
        """
        return await self._process_node(skeleton, source_id, id_prefix)

    async def _process_node(
        self, node: SkeletonNode, source_id: str, node_id: str
    ) -> IndexNode:
        # Process children first (bottom-up), in parallel per level
        children: list[IndexNode] = []
        if node.children:
            tasks = [
                self._process_node(child, source_id, f"{node_id}.{i}")
                for i, child in enumerate(node.children)
            ]
            children = list(await asyncio.gather(*tasks))

        # Generate summary
        if node.is_leaf:
            summary, nav_summary = await self._summarize_leaf(node)
        else:
            summary, nav_summary = await self._summarize_branch(node.title, children)

        # Per-leaf source_id override (used by CrossSourceStructurer)
        effective_source_id = source_id
        if node.is_leaf and node.page and "original_source_id" in node.page.metadata:
            effective_source_id = node.page.metadata["original_source_id"]

        # Compute content hash for leaf nodes
        content_hash: str | None = None
        if node.is_leaf and node.page and node.page.content:
            import hashlib
            content_hash = hashlib.sha256(node.page.content.encode()).hexdigest()

        return IndexNode(
            id=node_id,
            title=node.title,
            summary=summary,
            nav_summary=nav_summary,
            ref=node.ref,
            ref_type=node.ref_type,
            source_id=effective_source_id,
            content_hash=content_hash,
            children=children,
        )

    async def _summarize_leaf(self, node: SkeletonNode) -> tuple[str, str]:
        """Summarize a leaf node from its page content."""
        # Use llms.txt description if available (significant speed-up)
        if node.page and node.page.metadata.get("llms_txt_description"):
            desc = node.page.metadata["llms_txt_description"]
            if len(desc) > 50:
                # If it's a decent length, reuse it to avoid an LLM call
                if len(desc) < 200:
                    return desc, desc[:100]
                # If it's very long, we'll still call the LLM to get a good navigational snippet
                # but we'll use the desc as the content instead of the full page.

        if not node.page or not node.page.content.strip():
            return f"Documentation page: {node.title}", f"Doc: {node.title}"

        content = node.page.content[:MAX_CONTENT_CHARS]
        resp_text = await self._call_llm(
            LEAF_SYSTEM_PROMPT,
            f"Page title: {node.title}\n\nContent:\n{content}",
        )
        return self._parse_summaries(resp_text)

    async def _summarize_branch(self, title: str, children: list[IndexNode]) -> tuple[str, str]:
        """Summarize an internal node from its children's titles and summaries."""
        children_info = "\n".join(
            f"- {child.title}: {child.nav_summary or child.summary}" for child in children
        )
        resp_text = await self._call_llm(
            BRANCH_SYSTEM_PROMPT,
            f"Section title: {title}\n\nChild sections:\n{children_info}",
        )
        return self._parse_summaries(resp_text)

    async def _call_llm(self, system: str, user_message: str) -> str:
        """Make an LLM call with concurrency limiting and retries."""
        async with self._semaphore:
            max_retries = 3
            last_err = None
            
            for attempt in range(max_retries):
                try:
                    provider = self._get_provider()
                    response = await provider.generate(
                        user_message=user_message,
                        system=system,
                        max_tokens=256,
                    )
                    return response.text
                except Exception as e:
                    last_err = e
                    # Exponential backoff for retries
                    wait_time = 2 ** attempt
                    logger.warning(
                        "LLM summarization attempt %d/%d failed: %s (retrying in %ds)", 
                        attempt + 1, max_retries, e or type(e).__name__, wait_time
                    )
                    await asyncio.sleep(wait_time)
            
            logger.error("All %d summarization attempts failed for: %s", max_retries, user_message[:100])
            return f"Documentation section: {user_message[:100]}"
