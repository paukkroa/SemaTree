"""Main experiment runner -- runs all questions through all systems and collects metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from sema_tree.llm import LLMProvider, get_provider
from evaluation.config import AgenticConfig, ExperimentConfig
from evaluation.corpus.fetcher import fetch_corpus
from evaluation.corpus.preprocessor import preprocess_corpus
from evaluation.dataset.generator import load_questions
from evaluation.scoring.cost_tracker import ExperimentCostTracker
from evaluation.scoring.quality_judge import judge_answer
from evaluation.scoring.retrieval import compute_all_retrieval_metrics
from evaluation.systems.agentic_system import SemaTreeSystem
from evaluation.systems.base import RetrievalResult, RetrievalSystem

logger = logging.getLogger(__name__)


def _rag_available() -> bool:
    """Check whether the RAG baseline dependencies are usable."""
    try:
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401
        import rank_bm25 # noqa: F401
    except ImportError:
        return False
    return True


async def _run_single_query(
    system: RetrievalSystem,
    question: dict,
    trial: int,
    judge_provider: LLMProvider | None = None,
) -> dict:
    """Run a single question through a system and score the result."""
    q_text = question["question"]
    gold_answer = question["gold_answer"]
    gold_sources = question["gold_sources"]

    result: RetrievalResult = await system.retrieve(q_text)

    # Retrieval metrics
    retrieval_metrics = compute_all_retrieval_metrics(gold_sources, result.retrieved_sources)

    # Quality judge
    if judge_provider:
        try:
            judge_scores = await judge_answer(
                question=q_text,
                gold_answer=gold_answer,
                candidate_answer=result.answer,
                retrieved_content=result.retrieved_content,
                provider=judge_provider,
            )
            quality_scores = judge_scores.to_dict()
        except Exception as e:
            logger.error("Judge scoring failed for %s trial %d: %s", question["id"], trial, e)
            quality_scores = {
                "correctness": 0, "completeness": 0, "relevance": 0, "groundedness": 0, "composite": 0.0,
            }
    else:
        quality_scores = {
            "correctness": 0, "completeness": 0, "relevance": 0, "groundedness": 0, "composite": 0.0,
        }

    return {
        "question_id": question["id"],
        "category": question["category"],
        "system": system.name,
        "trial": trial,
        "answer": result.answer,
        "retrieved_sources": result.retrieved_sources,
        "retrieved_content": result.retrieved_content,
        "latency_ms": result.latency_ms,
        "tokens_used": result.tokens_used,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "embedding_tokens": result.embedding_tokens,
        "api_calls": result.api_calls,
        "tool_counts": result.tool_counts,
        **retrieval_metrics,
        **quality_scores,
    }


async def _build_systems(
    config: ExperimentConfig,
    pages,
    provider: LLMProvider,
    *,
    skip_rag: bool = False,
    current_index: str = "my_knowledge_base",
) -> list[RetrievalSystem]:
    """Instantiate and set up all retrieval systems."""
    systems: list[RetrievalSystem] = []

    if not skip_rag:
        for rag_cfg in config.rag_configs:
            from evaluation.systems.rag_baseline import RAGBaseline
            system = RAGBaseline(rag_cfg, pages, provider=provider)
            await system.setup()
            systems.append(system)
    else:
        logger.info("Skipping RAG baseline")

    for ag_cfg in config.agentic_configs:
        system = SemaTreeSystem(ag_cfg, current_index, provider=provider)
        await system.setup()
        systems.append(system)

    return systems


async def run_experiment(
    config: ExperimentConfig | None = None,
    provider: LLMProvider | None = None,
    skip_rag: bool = False,
    skip_judge: bool = False,
    question_limit: int | None = None,
    run_scaled: bool = False,
    index_override: str | None = None,
) -> dict:
    """Run the evaluation experiment."""
    config = config or ExperimentConfig()
    provider = provider or get_provider()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which indices to run
    if index_override:
        indices = [index_override]
    elif run_scaled:
        indices = ["my_knowledge_base_scaled"]
    else:
        indices = ["my_knowledge_base"]
        
    all_records: list[dict] = []
    all_cost_summaries = {}

    # Load questions once
    print(f"[1/7] Loading questions...")
    questions = load_questions(Path(config.questions_path))
    if question_limit:
        questions = questions[:question_limit]
    print(f"      Loaded {len(questions)} questions")

    for index_path_str in indices:
        index_path = Path(index_path_str)
        if not index_path.exists():
            print(f"Skipping missing index: {index_path_str}")
            continue

        print(f"\n--- Processing Index: {index_path_str} ---")
        
        # 1. Index files from the current folder
        print(f"      Indexing {index_path_str}...")
        md_files = list(index_path.rglob("*.md"))
        pages = preprocess_corpus(md_files, root_dir=index_path)
        print(f"      Indexed {len(pages)} documents")

        # 2. Build systems for THIS index
        systems = await _build_systems(config, pages, provider, skip_rag=skip_rag, current_index=index_path_str)
        
        total_queries = len(questions) * config.trials * len(systems)
        completed = 0

        for system in systems:
            display_name = f"{system.name}[{index_path.name}]"
            tracker = ExperimentCostTracker(system_name=display_name)
            
            print(f"      System: {display_name}")
            for qi, question in enumerate(questions):
                for trial in range(config.trials):
                    completed += 1
                    print(f"      [{completed}/{total_queries}] Q={question['id']} trial={trial+1}", end="", flush=True)
                    
                    record = await _run_single_query(
                        system,
                        question,
                        trial + 1,
                        judge_provider=None if skip_judge else provider,
                    )
                    record["system"] = display_name
                    record["index_type"] = index_path.name
                    all_records.append(record)
                    print(f"  latency={record['latency_ms']:.0f}ms  comp={record['composite']}")

                    tracker.add_query(
                        model=record.get("model", "unknown"),
                        input_tokens=record["input_tokens"],
                        output_tokens=record["output_tokens"],
                    )
            
            all_cost_summaries[display_name] = tracker.summary()
            await system.teardown()

    # 7. Aggregate and Save
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "trials": config.trials,
            "num_questions": len(questions),
            "indices": indices,
        },
        "records": all_records,
        "cost_summaries": all_cost_summaries,
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n[7/7] All results saved to {results_path}")
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SemaTree evaluation.")
    parser.add_argument("--index", type=str, default="", help="Path override for index.")
    parser.add_argument("--provider", type=str, default="auto", choices=["auto", "ollama", "gemini"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--fast", action="store_true", default=False, help="Fast mode: 5 Qs, 1 Trial")
    parser.add_argument("--scale-test", action="store_true", default=False, help="Run both normal and scaled indices")
    parser.add_argument("--skip-rag", action="store_true", default=False)
    parser.add_argument("--no-judge", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)

    provider = get_provider(provider=args.provider, model=args.model)
    print(f"Using LLM provider: {provider!r}")

    # Build experiment config
    default_config = ExperimentConfig()
    exp_kwargs: dict = {
        "agentic_configs": default_config.agentic_configs,
    }

    if args.trials is not None:
        exp_kwargs["trials"] = args.trials
    elif args.fast:
        exp_kwargs["trials"] = 1

    config = ExperimentConfig(**exp_kwargs)

    asyncio.run(
        run_experiment(
            config, 
            provider=provider, 
            skip_rag=args.skip_rag, 
            skip_judge=args.no_judge,
            question_limit=5 if args.fast else None,
            run_scaled=args.scale_test,
            index_override=args.index if args.index else None
        )
    )


if __name__ == "__main__":
    main()
