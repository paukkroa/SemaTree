# Agentic Index Evaluation Report

## 1. Executive Summary
This report evaluates **Agentic Index**, a hierarchical AI-navigable document structure, against traditional **Hybrid RAG** systems. Through a rigorous human-in-the-loop study of 640+ Q&A pairs, we demonstrate that agentic navigation fundamentally solves the "noise" and "needle-in-haystack" issues of vector search. While automated metrics (Keyword Recall) favor the "broad retrieval" approach of RAG, human evaluation confirms that Agentic Index provides superior precision and completeness for complex technical documentation.

---

## 2. Methodology & Scientific Transparency

### 2.1 Study Design
To ensure scientific validity and minimize bias, the study followed these core principles:
*   **Blind Evaluation**: The human judge used a custom-built FastAPI application where system names and trial indicators were completely anonymized.
*   **Randomization**: Side-by-side A/B preference tests were performed with randomized ordering (swapping System A and B) to prevent positional bias.
*   **Sequential Scoring**: For metric consistency, the judge scored "Deck Mode" cards sorted by Question ID. This allowed the judge to maintain a consistent "Gold Answer" mental model across all system attempts for a single question.
*   **Comparative Baseline**: We compared three Agentic strategies (Explicit, Navigational, Simplified) against a state-of-the-art Hybrid RAG baseline (Vector + Keyword search).

### 2.2 Datasets
Two distinct corpora were used to test scalability:
1.  **Normal (Targeted)**: 55 documents (Claude Code documentation).
2.  **Scaled (Noise-Rich)**: 450+ documents, adding OpenAI Codex docs, Python Library docs, and Purina dog breed data to simulate a large, heterogeneous enterprise knowledge base.

### 2.3 Evaluation Metrics
We utilized a multi-layered metric suite:
1.  **Human Quality (Ground Truth)**:
    *   **Precision (Correctness)**: 1-5 scale of technical accuracy.
    *   **Recall (Completeness)**: 1-5 scale of how many required steps/details were included.
2.  **Automated Accuracy**:
    *   **Keyword Recall**: Fraction of technical keywords from the gold answer present in the system answer.
    *   **Semantic Similarity**: Cosine similarity using the `all-MiniLM-L6-v2` transformer model.
3.  **Efficiency**: API calls, token consumption, and latency.

---

## 3. Conceptual Approach: The Semantic File System

Agentic Index fundamentally rethinks retrieval as **Logical Navigation** rather than **Statistical Matching**. 

### 3.1 Comparison with Current Methods

| Method | Key Issues | Agentic Index Solution |
| :--- | :--- | :--- |
| **RAG** | **Needle in Haystack**: High noise chunks saturate context. **Similarity Issues**: "Sounds relevant" but is technically wrong (e.g. wrong version). | **Precise Pathfinding**: Agent identifies the *exact* file first, then reads only that content. No context saturation. |
| **Agentic Websearch** | **Misleading URLs**: Raw URL structures are often circular or non-descriptive. **Travesal Cost**: Agents waste calls following dead-end links. | **Semantic Paths**: Replaces raw URLs with a clean, AI-structured folder hierarchy (e.g. `/security/permissions` vs `site.com/docs/v2/ref/h-723x`). |
| **llms.txt** | **Narrow Context**: Flat, limited perspective. **Stale**: Difficult to maintain for dynamic or large codebases. | **Recursive Summarization**: Every folder has a "Summary-of-Summaries," providing a map that is automatically generated and hierarchically deep. |
| **PageIndex / TOC** | **TOC Dependency**: Fails if the site's Table of Contents is poor. **Single-Doc focus**: Weak at synthesizing across multiple files. | **Autonomous Structuring**: Indexer uses LLMs to *re-group* pages logically, independent of the original site's navigation structure. |

---

## 4. System Architecture & Implementation

### 4.1 The Indexing Pipeline
1.  **Crawl**: Ingests flat Markdown/HTML from web or local sources.
2.  **Structure**: An LLM analyzes all page titles/metadata to build a "Skeleton Tree." It ignores raw URL structures and creates a human-logical hierarchy (e.g., grouping all "Cloud Providers" under a single node).
3.  **Summarize**: A bottom-up recursive process. Leaf nodes (documents) get a technical summary. Parent nodes (folders) get a **Summary-of-Summaries**, allowing an agent to understand an entire branch of the tree without entering it.
4.  **Assemble**: The tree is persisted as a physical directory structure.

### 4.2 The Navigable Tree Structure
*   **Directories**: Represent conceptual domains. Each contains a `_summary.md` (the "Map").
*   **Markdown Files**: The actual documents, containing AI-generated technical summaries and original source metadata (`ref` URLs).
*   **Logical vs. Physical**: The agent interacts with a virtualized version of this tree via the Model Context Protocol (MCP).

### 4.3 Tool Interaction Design
*   **`ls(path)`**: The agent's "eyes." It shows what's in a folder. Optimized modes (Navigational) inject micro-summaries into the output, allowing "Zero-Call" discovery of sub-paths.
*   **`find(pattern)`**: The agent's "teleporter." A global search that returns paths, allowing the agent to jump across the hierarchy if it has a specific keyword in mind.
*   **`get_details(path)`**: The "truth" tool. It fetches the *full, live* technical content. This ensures the final answer is based on the actual source, not an AI summary.

---

## 5. Hypothesis Testing Results

### H1: AgenticIndex retrieves gold standard documents more often (Precision)
**Status: CONFIRMED**  
Agentic Index (Explicit) retrieves only **1.17 sources** per query to achieve ~4.35 correctness. Hybrid RAG requires **3.9 to 4.2 sources** (mostly noise) to reach the same conclusion.

### H2: AgenticIndex retrieves less noise than Hybrid RAG
**Status: CONFIRMED**  
RAG systems retrieve **3.5x more distractor content**. Agentic Index's "logical navigation" prevents "hallucination by context saturation" common in chunk-based retrieval.

### H3: AgenticIndex provides more complete answers (Completeness)
**Status: CONFIRMED (Category Specific)**  
For **Procedural (PROC)** and **Comparison (COMP)** tasks, completeness scored **4.15/5**. Retrieving entire files allows the agent to synthesize sequences that are impossible when working with disconnected vector chunks.

### H4: Simple AgenticIndex (2 tools) leads to performance gains
**Status: REJECTED**  
The **Simplified** strategy performed worst (**3.25**). The dedicated "Peek" tool (`get_summary`) is critical for efficient navigation.

### H5: AgenticIndex quality decreases less than Hybrid RAG as scale increases
**Status: CONFIRMED**  
Agentic Index correctness remained stable at **~4.35** across both Normal and Scaled datasets. Hierarchy shields the agent from "haystack" growth.

### H6: Micro-summaries in `ls` provide more efficient search
**Status: INCONCLUSIVE**  
Summaries increased tool calls (**7.1 vs 6.4**). They provide more context but invite more exploration of distractor paths.

---

## 6. Metric Correlation Analysis

| Metric Pair | Correlation ($r$) | Interpretation |
| :--- | :--- | :--- |
| **Correctness vs. Keyword Recall** | **0.42** | Moderate. Keywords are a "noisy" signal for accuracy. |
| **Correctness vs. Semantic Similarity** | **0.29** | Low. Fluency does not imply technical truth. |
| **Keyword Recall vs. Semantic Similarity** | **0.80** | High. Both metrics largely measure word-overlap style. |

**Final Scientific Conclusion**: Automated metrics significantly over-reward RAG systems for "word spraying." Human evaluation confirms that Agentic Index's **Logical Navigation** paradigm provides a more reliable, precise, and scalable architecture for technical knowledge retrieval.
