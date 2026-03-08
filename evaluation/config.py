"""Configuration constants for the evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_ROOT.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
CORPUS_CACHE_DIR = CACHE_DIR / "corpus"
RESULTS_DIR = EVAL_ROOT / "results"

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
CORPUS_URL = "https://code.claude.com/docs/en"
CORPUS_SITEMAP_URLS: list[str] = [
    "https://code.claude.com/docs/en/overview",
    "https://code.claude.com/docs/en/getting-started",
    "https://code.claude.com/docs/en/quickstart",
    "https://code.claude.com/docs/en/setup",
    "https://code.claude.com/docs/en/how-claude-code-works",
    "https://code.claude.com/docs/en/cli-usage",
    "https://code.claude.com/docs/en/memory",
    "https://code.claude.com/docs/en/settings",
    "https://code.claude.com/docs/en/permissions",
    "https://code.claude.com/docs/en/sandboxing",
    "https://code.claude.com/docs/en/terminal-config",
    "https://code.claude.com/docs/en/model-config",
    "https://code.claude.com/docs/en/fast-mode",
    "https://code.claude.com/docs/en/keybindings",
    "https://code.claude.com/docs/en/statusline",
    "https://code.claude.com/docs/en/ide-integrations",
    "https://code.claude.com/docs/en/vs-code",
    "https://code.claude.com/docs/en/jetbrains",
    "https://code.claude.com/docs/en/github-actions",
    "https://code.claude.com/docs/en/gitlab-ci-cd",
    "https://code.claude.com/docs/en/slack",
    "https://code.claude.com/docs/en/chrome",
    "https://code.claude.com/docs/en/claude-code-on-the-web",
    "https://code.claude.com/docs/en/mcp-servers",
    "https://code.claude.com/docs/en/mcp",
    "https://code.claude.com/docs/en/security",
    "https://code.claude.com/docs/en/troubleshooting",
    "https://code.claude.com/docs/en/costs",
    "https://code.claude.com/docs/en/analytics",
    "https://code.claude.com/docs/en/data-usage",
    "https://code.claude.com/docs/en/monitoring-usage",
    "https://code.claude.com/docs/en/authentication",
    "https://code.claude.com/docs/en/server-managed-settings",
    "https://code.claude.com/docs/en/enterprise-deployment-overview",
    "https://code.claude.com/docs/en/amazon-bedrock",
    "https://code.claude.com/docs/en/google-vertex-ai",
    "https://code.claude.com/docs/en/microsoft-foundry",
    "https://code.claude.com/docs/en/llm-gateway",
    "https://code.claude.com/docs/en/network-config",
    "https://code.claude.com/docs/en/devcontainer",
    "https://code.claude.com/docs/en/agent-teams",
    "https://code.claude.com/docs/en/sub-agents",
    "https://code.claude.com/docs/en/plugins",
    "https://code.claude.com/docs/en/discover-plugins",
    "https://code.claude.com/docs/en/plugin-marketplaces",
    "https://code.claude.com/docs/en/skills",
    "https://code.claude.com/docs/en/hooks-guide",
    "https://code.claude.com/docs/en/hooks",
    "https://code.claude.com/docs/en/output-styles",
    "https://code.claude.com/docs/en/headless",
    "https://code.claude.com/docs/en/best-practices",
    "https://code.claude.com/docs/en/common-workflows",
    "https://code.claude.com/docs/en/checkpointing",
    "https://code.claude.com/docs/en/interactive-mode",
    "https://code.claude.com/docs/en/cli-reference",
    "https://code.claude.com/docs/en/plugins-reference",
    "https://code.claude.com/docs/en/hooks-reference",
]

# ---------------------------------------------------------------------------
# RAG configuration space
# ---------------------------------------------------------------------------
CHUNK_SIZES: list[int] = [256, 512, 1024]
CHUNK_OVERLAP_RATIO: float = 0.1
TOP_K_VALUES: list[int] = [3, 5, 10]

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # local sentence-transformers model
GENERATION_MODEL = "gemini-3-flash-preview"
JUDGE_MODEL = "gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# Pricing (USD per 1 K tokens, as of 2025-05)
# ---------------------------------------------------------------------------
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-3-flash-preview": {"input": 0.00015, "output": 0.0006},
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
    "all-MiniLM-L6-v2": {"input": 0.0, "output": 0.0},  # local, free
}

# ---------------------------------------------------------------------------
# Experiment defaults
# ---------------------------------------------------------------------------
DEFAULT_TRIALS = 2
QUESTION_CATEGORIES = ["SF", "MH", "COMP", "PROC"]

# ---------------------------------------------------------------------------
# System configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RAGConfig:
    """Configuration for a RAG baseline run."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    embedding_model: str = EMBEDDING_MODEL
    generation_model: str = GENERATION_MODEL
    collection_name: str = "eval_corpus"

    @property
    def label(self) -> str:
        return f"RAG(chunk={self.chunk_size}, k={self.top_k})"


@dataclass
class AgenticConfig:
    """Configuration for an SemaTree run."""

    generation_model: str = GENERATION_MODEL
    index_path: str = ""
    max_exploration_depth: int = 5
    max_fetches: int = 10
    strategy: str = "explicit"  # "explicit" (3 tools), "simplified" (2 tools), or "navigational" (ls+snippets)

    @property
    def label(self) -> str:
        return f"Agentic-{self.strategy}(depth={self.max_exploration_depth}, fetches={self.max_fetches})"


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    rag_configs: list[RAGConfig] = field(default_factory=lambda: [RAGConfig()])
    agentic_configs: list[AgenticConfig] = field(default_factory=lambda: [
        AgenticConfig(strategy="explicit"),
        AgenticConfig(strategy="simplified"),
        AgenticConfig(strategy="navigational"),
    ])
    trials: int = DEFAULT_TRIALS
    questions_path: str = str(EVAL_ROOT / "dataset" / "questions.json")
    output_dir: str = str(RESULTS_DIR)
