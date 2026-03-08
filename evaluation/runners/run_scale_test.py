"""Scalability test -- measures performance at different corpus sizes."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from evaluation.config import (
    RESULTS_DIR,
    RAGConfig,
)
from evaluation.corpus.fetcher import fetch_corpus
from evaluation.corpus.preprocessor import DocPage, preprocess_corpus
from evaluation.dataset.generator import load_questions
from evaluation.scoring.retrieval import compute_all_retrieval_metrics
from evaluation.systems.base import RetrievalResult
from sema_tree.llm import LLMProvider, get_provider

logger = logging.getLogger(__name__)

SCALE_SIZES = [56, 200, 500]


def _rag_available() -> bool:
    """Check whether the RAG baseline dependencies (chromadb, sentence-transformers) are usable."""
    try:
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _simulate_corpus_at_size(pages: list[DocPage], target_size: int) -> list[DocPage]:
    """Simulate a larger corpus by duplicating pages with modified slugs.

    If ``target_size`` is smaller than the real corpus, take a subset.
    """
    if target_size <= len(pages):
        return pages[:target_size]

    # Duplicate pages to reach target size
    expanded: list[DocPage] = []
    idx = 0
    while len(expanded) < target_size:
        src = pages[idx % len(pages)]
        copy_num = idx // len(pages)
        expanded.append(
            DocPage(
                path=src.path,
                slug=f"{src.slug}_dup{copy_num}" if copy_num > 0 else src.slug,
                title=src.title,
                content=src.content,
                headings=src.headings,
                word_count=src.word_count,
            )
        )
        idx += 1
    return expanded


async def run_scale_test(
    questions_path: Path | None = None,
    output_dir: Path | None = None,
    sizes: list[int] | None = None,
    provider: LLMProvider | None = None,
) -> dict:
    """Run the scalability test across different corpus sizes.

    Args:
        questions_path: Path to questions JSON.
        output_dir: Directory for results output.
        sizes: Corpus sizes to test.
        provider: LLM provider instance. Resolved via ``get_provider()`` when *None*.
    """
    sizes = sizes or SCALE_SIZES
    output_dir = output_dir or RESULTS_DIR / "scale_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = provider or get_provider()
    print(f"Using LLM provider: {provider!r}")

    if not _rag_available():
        print("[warning] Scale test requires chromadb + OPENAI_API_KEY for the RAG baseline.")
        print("          Skipping scale test.")
        return {}

    from evaluation.systems.rag_baseline import RAGBaseline

    # Load base corpus
    print("[1/3] Fetching corpus...")
    cached_paths = await fetch_corpus()
    base_pages = preprocess_corpus(cached_paths)
    print(f"      Base corpus: {len(base_pages)} pages")
    logger.info("Base corpus: %d pages", len(base_pages))

    # Load questions
    qpath = questions_path or Path(__file__).resolve().parent.parent / "dataset" / "questions.json"
    questions = load_questions(qpath)
    print(f"      Loaded {len(questions)} questions")

    rag_config = RAGConfig()
    results_by_size: dict[int, dict] = {}

    print(f"[2/3] Running scale tests for sizes: {sizes}")
    for si, size in enumerate(sizes):
        print(f"      === Scale {si + 1}/{len(sizes)}: {size} pages ===")
        logger.info("=== Scale test: %d pages ===", size)
        pages = _simulate_corpus_at_size(base_pages, size)

        system = RAGBaseline(rag_config, pages, provider=provider)
        await system.setup()

        f1_scores: list[float] = []
        latencies: list[float] = []
        costs: list[float] = []

        for qi, question in enumerate(questions):
            print(
                f"        Q {qi + 1}/{len(questions)}: {question['id']}",
                end="",
                flush=True,
            )
            result: RetrievalResult = await system.retrieve(question["question"])
            metrics = compute_all_retrieval_metrics(
                question["gold_sources"], result.retrieved_sources
            )
            f1_scores.append(metrics["f1"])
            latencies.append(result.latency_ms)
            costs.append(result.tokens_used)
            print(f"  F1={metrics['f1']:.3f}  latency={result.latency_ms:.0f}ms")

        await system.teardown()

        results_by_size[size] = {
            "corpus_size": size,
            "mean_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)),
            "mean_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "mean_tokens": float(np.mean(costs)),
            "num_questions": len(questions),
        }
        print(
            f"      Summary: F1={results_by_size[size]['mean_f1']:.3f} "
            f"(+-{results_by_size[size]['std_f1']:.3f}), "
            f"Latency={results_by_size[size]['mean_latency_ms']:.0f}ms"
        )
        logger.info(
            "  F1=%.3f (+-%.3f), Latency=%.0fms",
            results_by_size[size]["mean_f1"],
            results_by_size[size]["std_f1"],
            results_by_size[size]["mean_latency_ms"],
        )

    # Save results
    out_path = output_dir / "scale_results.json"
    out_path.write_text(json.dumps(results_by_size, indent=2), encoding="utf-8")
    print(f"[3/3] Scale test results saved to {out_path}")
    logger.info("Scale test results saved to %s", out_path)

    return results_by_size


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SemaTree scalability test.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "ollama", "gemini"],
        help="LLM provider to use (default: auto-detect).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override for the chosen provider.",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Path to the evaluation questions JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to write scale test results into.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="Corpus sizes to test (default: 56 200 500).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)

    provider = get_provider(provider=args.provider, model=args.model)

    questions_path = Path(args.questions) if args.questions else None
    output_dir = Path(args.output) if args.output else None

    asyncio.run(
        run_scale_test(
            questions_path=questions_path,
            output_dir=output_dir,
            sizes=args.sizes,
            provider=provider,
        )
    )


if __name__ == "__main__":
    main()
