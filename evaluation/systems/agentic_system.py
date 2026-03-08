"""SemaTree retrieval system — actual MCP client."""

from __future__ import annotations

import os
import re
import time
import json
import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from sema_tree.llm import LLMProvider, get_provider
from evaluation.config import AgenticConfig
from evaluation.systems.base import RetrievalResult, RetrievalSystem


class SemaTreeSystem(RetrievalSystem):
    """Retrieval using an actual MCP client connection.

    Simulates a real-world agent connecting to the documentation MCP server
    over stdio.
    """

    def __init__(
        self,
        config: AgenticConfig,
        index_path: str,
        provider: LLMProvider | None = None,
    ) -> None:
        self._config = config
        self._index_path = os.path.abspath(index_path)
        self._provider = provider or get_provider()
        
        # Build server parameters for stdio
        self._server_params = StdioServerParameters(
            command="uv",
            args=["run", "sema-tree", "serve", self._index_path],
            env=os.environ.copy()
        )

    @property
    def name(self) -> str:
        return self._config.label

    async def setup(self) -> None:
        """No pre-setup needed, connection is per-query."""

    async def retrieve(self, question: str) -> RetrievalResult:
        """Execute the agentic loop via a live MCP connection."""
        t0 = time.perf_counter()
        total_input_tokens = 0
        total_output_tokens = 0
        api_calls = 0
        tool_counts = {"ls": 0, "get_summary": 0, "get_details": 0, "find": 0}
        retrieved_sources: list[str] = []
        
        # Strategy-specific system prompts
        if self._config.strategy == "simplified":
            system_prompt = (
                "You are the Assistant, a documentation expert with access to an MCP server.\n"
                "The conversation history shows YOUR own previous thoughts and actions.\n\n"
                "Tool Usage Guidelines:\n"
                "1. ALWAYS use absolute paths starting with `/` (e.g., TOOL: ls(path='/')).\n"
                "2. CRITICAL: Only use paths explicitly listed in the output of the `ls` or `find` tools.\n"
                "3. `ls(path, depth=N)`: Use this to explore the document hierarchy.\n"
                "4. `find(pattern)`: Search for document filenames or directory names matching a pattern.\n"
                "5. `read(path)`: Retrieves the FULL, LATEST technical content of a document. ALWAYS use this before giving a final answer.\n\n"
                "Instructions:\n"
                "1. Always think step-by-step. Use 'Thought: <reasoning>'.\n"
                "2. To use a tool, output exactly: TOOL: name(arg='value')\n"
                "3. When you have the final answer, use TOOL: answer(text='...')\n"
                "4. Keep responses concise and helpful."
            )
            tool_map = {"read": "get_details", "ls": "ls", "find": "find", "answer": "answer"}
        elif self._config.strategy == "navigational":
            system_prompt = (
                "You are the Assistant, a documentation expert with access to an MCP server.\n"
                "The conversation history shows YOUR own previous thoughts and actions.\n\n"
                "Tool Usage Guidelines:\n"
                "1. ALWAYS use absolute paths starting with `/`.\n"
                "2. `ls(path, depth=N)`: Explore the document hierarchy. This tool returns concise summaries for each document automatically to help you find the right one.\n"
                "3. `find(pattern)`: Search for document filenames or directory names matching a pattern.\n"
                "4. `read(path)`: Retrieves the FULL, LATEST technical content. Use this once you've found the relevant file via `ls` or `find`.\n\n"
                "Instructions:\n"
                "1. Always think step-by-step. Use 'Thought: <reasoning>'.\n"
                "2. To use a tool, output exactly: TOOL: name(arg='value')\n"
                "3. When you have the final answer, use TOOL: answer(text='...')\n"
                "4. Keep responses concise and helpful."
            )
            tool_map = {"read": "get_details", "ls": "ls", "find": "find", "answer": "answer"}
        else:
            system_prompt = (
                "You are the Assistant, a documentation expert with access to an MCP server.\n"
                "The conversation history shows YOUR own previous thoughts and actions.\n\n"
                "Tool Usage Guidelines:\n"
                "1. ALWAYS use absolute paths starting with `/`.\n"
                "2. `ls(path, depth=N)`: Explore the document hierarchy.\n"
                "3. `find(pattern)`: Search for document filenames or directory names matching a pattern.\n"
                "4. `get_summary(path)`: Provides a BRIEF overview. Use this to check if a file is relevant.\n"
                "5. `get_details(path)`: Provides the FULL technical content. ALWAYS use this before answering technical questions.\n\n"
                "Instructions:\n"
                "1. Always think step-by-step. Use 'Thought: <reasoning>'.\n"
                "2. To use a tool, output exactly: TOOL: name(arg='value')\n"
                "3. When you have the final answer, use TOOL: answer(text='...')\n"
                "4. Keep responses concise and helpful."
            )
            tool_map = {"get_summary": "get_summary", "get_details": "get_details", "ls": "ls", "find": "find", "answer": "answer"}

        history = [
            f"User: {question}\n"
            "Explore the documentation to find the answer. Start by listing the root."
        ]

        final_answer = ""
        # Dynamic turn limit based on config
        max_turns = self._config.max_exploration_depth + self._config.max_fetches + 2

        # Start live MCP session
        async with stdio_client(self._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                for _ in range(max_turns):
                    prompt = "\n".join(history) + "\nAssistant:"
                    
                    try:
                        resp = await self._provider.generate(
                            user_message=prompt,
                            system=system_prompt,
                            max_tokens=1024,
                            temperature=0.0
                        )
                    except Exception as e:
                        history.append(f"System: [LLM Error: {e}]")
                        break
                    
                    api_calls += 1
                    total_input_tokens += resp.input_tokens
                    total_output_tokens += resp.output_tokens
                    
                    assistant_text = resp.text.strip()
                    if resp.thought:
                        history.append(f"Assistant Thought: {resp.thought}")
                    history.append(f"Assistant: {assistant_text}")

                    # Check for TOOL: answer
                    ans_match = re.search(r"TOOL:\s*answer\(text=['\"](.*?)['\"]\)", assistant_text, re.DOTALL)
                    if ans_match:
                        final_answer = ans_match.group(1).strip()
                        break

                    # Parse Tool Call
                    match = re.search(r"TOOL:\s*(\w+)\((.*?)\)", assistant_text)
                    if not match:
                        history.append(f"System: Please use a tool ({', '.join(tool_map.keys())}, or answer).")
                        continue

                    tool_name_raw = match.group(1).lower()
                    tool_args_raw = match.group(2).strip()

                    if tool_name_raw not in tool_map:
                        history.append(f"System: Unknown tool '{tool_name_raw}'. Available: {', '.join(tool_map.keys())}")
                        continue

                    tool_name = tool_map[tool_name_raw]

                    # Simple arg parsing
                    kwargs = {}
                    path_match = re.search(r"path=['\"](.*?)['\"]", tool_args_raw)
                    if path_match: kwargs["path"] = path_match.group(1)
                    
                    pattern_match = re.search(r"pattern=['\"](.*?)['\"]", tool_args_raw)
                    if pattern_match: kwargs["pattern"] = pattern_match.group(1)
                    
                    depth_match = re.search(r"depth=(\d+)", tool_args_raw)
                    if depth_match: kwargs["depth"] = int(depth_match.group(1))
                    
                    # ENHANCEMENT: Internally inject include_summaries for navigational strategy
                    if self._config.strategy == "navigational" and tool_name == "ls":
                        kwargs["include_summaries"] = True
                        
                    if "path" not in kwargs: kwargs["path"] = "/"

                    try:
                        # CALL THE ACTUAL MCP TOOL
                        result = await session.call_tool(tool_name, arguments=kwargs)
                        observation = "".join([c.text for c in result.content if hasattr(c, 'text')])
                        
                        # Metrics tracking
                        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                        if tool_name in ["get_summary", "get_details"]:
                            retrieved_sources.append(kwargs.get("path", "/"))
                        
                        # Truncate for prompt history
                        history_obs = observation
                        if len(history_obs) > 2000:
                            history_obs = history_obs[:2000] + "\n...[truncated]"
                            
                        history.append(f"System (Tool Result - {tool_name_raw}):\n{history_obs}")
                        
                    except Exception as e:
                        history.append(f"System: [MCP Tool Error: {e}]")

        if not final_answer:
            final_answer = "Could not find answer within step limit."

        latency_ms = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            retrieved_sources=retrieved_sources,
            retrieved_content="\n\n".join(history),
            answer=final_answer,
            tokens_used=total_input_tokens + total_output_tokens,
            latency_ms=latency_ms,
            api_calls=api_calls,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            embedding_tokens=0,
            model=self._config.generation_model,
            tool_counts=tool_counts,
        )

        if not final_answer:
            final_answer = "Could not find answer within step limit."

        latency_ms = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            retrieved_sources=retrieved_sources,
            retrieved_content="\n\n".join(history),
            answer=final_answer,
            tokens_used=total_input_tokens + total_output_tokens,
            latency_ms=latency_ms,
            api_calls=api_calls,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            embedding_tokens=0,
            model=self._config.generation_model,
        )
