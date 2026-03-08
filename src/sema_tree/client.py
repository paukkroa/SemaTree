"""Interactive ReAct-based MCP Chat Client for SemaTree."""

from __future__ import annotations

import asyncio
import os
import re
import sys
from typing import Any

import click
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from sema_tree.llm import LLMProvider, get_provider

# Basic ANSI colors
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"

class AgenticChat:
    """An interactive chat client that can use MCP tools as needed."""

    def __init__(
        self,
        server_params: StdioServerParameters,
        provider: LLMProvider,
    ):
        self.server_params = server_params
        self.provider = provider
        self.history: list[dict[str, str]] = []

    def _print_thought(self, thought: str | None):
        """Print the model's reasoning trace."""
        if thought:
            print(f"\n{CYAN}{BOLD}🤔 Thought:{RESET} {CYAN}{thought.strip()}{RESET}")

    async def start(self):
        """Start the interactive chat loop."""
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                tools = await session.list_tools()
                tool_list_str = "\n".join([f"- {t.name}: {t.description}" for t in tools.tools])

                system_prompt = (
                    "You are the Assistant, a helpful documentation expert with access to an MCP server.\n"
                    "The conversation history shows YOUR own previous thoughts and actions.\n\n"
                    f"Available Tools:\n{tool_list_str}\n\n"
                    "Tool Usage Guidelines:\n"
                    "1. ALWAYS use absolute paths starting with `/` (e.g., TOOL: ls(path='/')).\n"
                    "2. CRITICAL: Only use files and directories that you have seen in the output of the `ls` tool. DO NOT assume or hallucinate paths that were not explicitly listed.\n"
                    "3. If a directory or file was not in the `ls` result, it does NOT exist.\n"
                    "4. `ls(path, depth=N)`: Use this to explore. Use `depth=2` to see a broader map.\n"
                    "5. `get_summary(path)`: Provides a BRIEF SUMMARY only. Use this to confirm if a file is relevant.\n"
                    "6. `get_details(path)`: Provides the FULL, LATEST technical content. ALWAYS use this before giving a final answer to a technical question.\n\n"
                    "Instructions:\n"
                    "1. Always think step-by-step. Use 'Thought: <reasoning>'.\n"
                    "2. Format tool calls exactly as: TOOL: name(path='/absolute/path', depth=N)\n"
                    "3. Don't repeat yourself. If the history shows you already listed a directory or read a file, use that information.\n"
                    "4. If you have enough info, respond normally without a tool call.\n"
                    "5. Keep responses concise and helpful."
                )

                print(f"\n{MAGENTA}{BOLD}💬 SemaTree Chat Started!{RESET}")
                print(f"Type {BOLD}'exit'{RESET} or {BOLD}'quit'{RESET} to end session.\n")

                while True:
                    try:
                        user_input = input(f"{BOLD}You:{RESET} ").strip()
                    except (EOFError, KeyboardInterrupt):
                        break

                    if user_input.lower() in ["exit", "quit"]:
                        break
                    
                    if not user_input:
                        continue

                    self.history.append({"role": "user", "content": user_input})
                    
                    # Agent Loop (can take multiple internal turns for tool use)
                    max_agent_turns = 10
                    error_occurred = False
                    turn = 0
                    for turn in range(max_agent_turns):
                        # Construct prompt from history
                        full_prompt = ""
                        for msg in self.history:
                            role = "User" if msg["role"] == "user" else ("Assistant" if msg["role"] == "assistant" else "System")
                            full_prompt += f"{role}: {msg['content']}\n"
                        full_prompt += "Assistant:"

                        try:
                            resp = await self.provider.generate(
                                user_message=full_prompt,
                                system=system_prompt,
                                max_tokens=1024,
                                temperature=0.0
                            )
                        except Exception as e:
                            print(f"\n{BOLD}❌ [LLM Provider Error: {e}]{RESET}")
                            print("Please try again in a moment.\n")
                            error_occurred = True
                            break
                        
                        assistant_text = resp.text.strip()
                        self.history.append({"role": "assistant", "content": assistant_text})

                        # Show thoughts to user (using the structured field)
                        self._print_thought(resp.thought)

                        # Parse for tool call
                        match = re.search(r"TOOL:\s*(\w+)\((.*?)\)", assistant_text)
                        
                        if match:
                            tool_name = match.group(1).lower()
                            tool_args_raw = match.group(2).strip()
                            
                            kwargs = {}
                            path_match = re.search(r"path=['\"](.*?)['\"]", tool_args_raw)
                            if path_match: kwargs["path"] = path_match.group(1)
                            depth_match = re.search(r"depth=(\d+)", tool_args_raw)
                            if depth_match: kwargs["depth"] = int(depth_match.group(1))
                            if "path" not in kwargs: kwargs["path"] = "/"

                            print(f"  {YELLOW}🛠️  Using {tool_name}({kwargs})...{RESET}")
                            
                            try:
                                result = await session.call_tool(tool_name, arguments=kwargs)
                                observation = "".join([c.text for c in result.content if hasattr(c, 'text')])
                                
                                # Show result preview to user
                                obs_preview = observation
                                if len(obs_preview) > 500:
                                    obs_preview = obs_preview[:500] + "... [truncated]"
                                
                                print(f"  {GREEN}📝 Result:{RESET}\n{GREEN}{obs_preview}{RESET}\n")
                                
                                self.history.append({"role": "system", "content": f"Tool Result ({tool_name}):\n{observation}"})
                            except Exception as e:
                                error_msg = f"Error calling tool: {e}"
                                self.history.append({"role": "system", "content": error_msg})
                                print(f"  {BOLD}❌ [Agent Error: {e}]{RESET}")
                        else:
                            print(f"\n{BOLD}Assistant:{RESET} {assistant_text}\n")
                            break
                    
                    if not error_occurred and turn == max_agent_turns - 1:
                        print(f"\n{BOLD}Assistant:{RESET} [Reached turn limit.]\n")


@click.command()
@click.argument("index_path")
@click.option("--provider", default="auto", type=click.Choice(["auto", "ollama", "gemini"]))
@click.option("--model", default=None)
def main(index_path: str, provider: str, model: str | None):
    """Run an interactive agentic chat session."""
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "sema-tree", "serve", os.path.abspath(index_path)],
        env=os.environ.copy()
    )

    llm = get_provider(provider=provider, model=model)
    chat = AgenticChat(server_params, llm)
    
    asyncio.run(chat.start())


if __name__ == "__main__":
    main()
