# Evaluation

This directory contains the experiment that benchmarks SemaTree against Hybrid RAG for technical documentation retrieval.

---

## Purpose

The evaluation tests five hypotheses about whether a navigable semantic file tree (SemaTree) offers advantages over Hybrid RAG for technical documentation Q&A:

- **H1**: SemaTree retrieves the gold-standard documents more often (higher precision)
- **H2**: SemaTree retrieves less noise in large, heterogeneous knowledge bases
- **H3**: SemaTree provides more complete answers than Hybrid RAG
- **H4**: A simplified 2-tool SemaTree leads to performance gains vs the full 4-tool set
- **H5**: Injecting micro-summaries into `ls` results provides more efficient search (fewer tool calls) than listing names only

---

## Dataset

### 40 questions, 4 categories

A set of 40 questions and gold answers was created from the Claude Code documentation, spanning four task types:

| Category | Description |
|---|---|
| **SF** (Single-Fact) | Answer is a specific value found in one place |
| **MH** (Multi-Hop) | Requires finding a reference in one doc and navigating to a second doc |
| **COMP** (Comparison) | Requires visiting two distinct branches and synthesising a comparison |
| **PROC** (Procedural) | Requires a complete sequence of steps — fails if any step is omitted |

Questions are in `evaluation/dataset/questions.json`.

### Corpus — 450+ documents (`my_knowledge_base_scaled/`)

The experiment used a noise-rich corpus designed to simulate a large, heterogeneous enterprise knowledge base:

- **Claude Code documentation** — the main corpus, source of all 40 questions
- **OpenAI Codex documentation** — semantically similar noise (same domain, different product)
- **Python 3 standard library docs** — technical noise from a related field
- **Purina dog breed data** — pure off-domain noise

The corpus is committed to the repository at `my_knowledge_base_scaled/` so the experiment can be reproduced without re-crawling.

---

## Systems Tested

Three SemaTree toolset configurations were compared against a Hybrid RAG baseline:

| System | Tools | Description |
|---|---|---|
| **Explicit** | `ls`, `find`, `get_summary`, `get_details` | Agent fully controls context cost at each step |
| **Navigational** | same | `ls` injects navigational summaries automatically |
| **Simplified** | `ls`, `get_details` | No `get_summary` peek step |
| **Hybrid RAG** | — | ChromaDB (`all-MiniLM-L6-v2`, 512-token chunks, 50-token overlap) + BM25 (Okapi BM25), reranked with RRF (k=60), top-5 |

All agentic systems used a **ReAct agent running Qwen3:8b locally via Ollama** on an M1 Pro MacBook Pro.

---

## Evaluation Metrics

### Human evaluation (ground truth)
- **Correctness (Precision)**: 1–5 scale of technical accuracy
- **Completeness (Recall)**: 1–5 scale of how many required steps/details were included

The judge used a custom FastAPI application (`evaluation/analysis/judge_app.py`) where system names were fully anonymised and answer ordering was randomised per question to prevent positional bias. The app presents answers in a blind A/B card format. A "Deck Mode" sorts cards by Question ID so the judge can maintain a consistent gold-answer mental model across all system attempts for a single question.

**Failed runs were excluded from human scoring.** The judge app automatically filters out any record where the answer is `"Could not find answer within step limit."` (agent reached max turns) before presenting cards. This applied to both agentic systems and RAG baseline runs that failed to produce an answer. Human scores are stored as `NaN` for unrated rows and excluded via `mean()` rather than treated as zero, so failed runs do not drag down any system's average.

To launch the judge app:

```bash
uvicorn evaluation.analysis.judge_app:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

### Automated metrics
- **Keyword Recall**: Fraction of gold-answer technical keywords present in the system answer
- **Semantic Similarity**: Cosine similarity using `all-MiniLM-L6-v2`

### Efficiency
- API calls, token consumption, and latency per query

---

## Reproducing the Experiment

### Prerequisites

Install evaluation dependencies:

```bash
uv pip install -e ".[eval]"
```

Ensure Ollama is running with the Qwen3:8b model pulled:

```bash
ollama pull qwen3:8b
```

The corpus (`my_knowledge_base_scaled/`) is already in the repository — no re-crawling needed.

### Run

```bash
python -m evaluation.runners.run_evaluation \
    --index my_knowledge_base_scaled \
    --provider ollama \
    --model qwen3:8b
```

Results are written to `evaluation/results/results.json`.

To quickly verify your setup works before committing to a full run (5 questions, 1 trial):

```bash
python -m evaluation.runners.run_evaluation \
    --index my_knowledge_base_scaled \
    --provider ollama \
    --model qwen3:8b \
    --fast
```

### Other useful flags

| Flag | Description |
|---|---|
| `--skip-rag` | Skip the Hybrid RAG baseline (faster, agentic systems only) |
| `--no-judge` | Skip LLM-based quality scoring (automated metrics only) |
| `--trials N` | Override number of trials per question (default: 2) |

---

## Results

Raw results are in `evaluation/results/results.json`. Automated metrics are in `evaluation/results/accuracy_analysis_all.json` (failed runs excluded from all metrics — see below).

### Hypothesis outcomes

**H1 — CONFIRMED**: SemaTree (Explicit) retrieved only **1.17 sources per query** vs **3.9–4.2 for Hybrid RAG** (mostly noise). Logical navigation finds the exact document; RAG floods context with distractor chunks.

**H2 — CONFIRMED**: RAG retrieved **3.5x more distractor content**. SemaTree quality remained stable across corpus sizes; RAG degraded with scale as noise chunks saturated context.

**H3 — CONFIRMED (category-specific)**: For **PROC (procedural)** tasks, SemaTree improved **~18% in precision and ~19% in recall** vs Hybrid RAG. Retrieving entire files lets the agent synthesise complete step sequences; disconnected vector chunks miss "boring but vital" steps. For simple SF (single-fact) tasks, Hybrid RAG remained faster and equally accurate.

**H4 — REJECTED**: The **Simplified** toolset (no `get_summary`) performed worst. Removing the peek step forces the agent to either fetch full content speculatively or navigate blind. The `get_summary` call is critical.

**H5 — INCONCLUSIVE**: Injecting summaries into `ls` increased tool calls (7.1 vs 6.4). The richer context invited more exploration of distractor paths, suggesting summaries can act as distractors that lead agents down irrelevant branches.

---

## Automated metric results

Agentic systems had a 27–43% failure rate (agent reached max turns); Hybrid RAG had 0%. Failed runs are excluded from all metrics — both automated and human — so the comparison reflects successful runs only.

| System | Keyword Recall | Semantic Similarity | Failed / Total |
|---|---|---|---|
| Hybrid-RAG | 0.683 | 0.759 | 0/80 |
| Agentic-explicit | 0.564 | 0.778 | 34/80 |
| Agentic-simplified | 0.550 | 0.782 | 26/80 |
| Agentic-navigational | 0.517 | 0.747 | 22/80 |

Keyword recall favours RAG — it retrieves more text, which naturally covers more gold-answer keywords. Semantic similarity is effectively tied: agentic-explicit and agentic-simplified slightly exceed RAG, navigational sits just below.
