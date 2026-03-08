"""ChromaDB-based vector RAG baseline system."""

from __future__ import annotations

import logging
import asyncio
import time
import re
import httpx
import json
from pathlib import Path
from bs4 import BeautifulSoup
from markdownify import markdownify

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from sema_tree.llm import LLMProvider, get_provider
from evaluation.config import RAGConfig
from evaluation.corpus.preprocessor import DocPage
from evaluation.systems.base import RetrievalResult, RetrievalSystem

logger = logging.getLogger(__name__)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by approximate word count."""
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


class RAGBaseline(RetrievalSystem):
    """Modern Hybrid RAG pipeline using Vector Search + BM25 merged via RRF."""

    def __init__(
        self,
        config: RAGConfig,
        pages: list[DocPage],
        provider: LLMProvider | None = None,
    ) -> None:
        self._config = config
        self._pages = pages
        self._provider = provider or get_provider()
        self._embedder: SentenceTransformer | None = None
        self._collection: chromadb.Collection | None = None
        self._client: chromadb.ClientAPI | None = None
        
        # Hybrid search components
        self._bm25: BM25Okapi | None = None
        self._all_chunks: list[str] = []
        self._all_metadatas: list[dict[str, str]] = []

    @property
    def name(self) -> str:
        return f"Hybrid-{self._config.label}"

    async def setup(self) -> None:
        """Build the vector and BM25 indices."""
        logger.info("Setting up Hybrid RAG baseline...")
        self._embedder = SentenceTransformer(self._config.embedding_model)
        self._client = chromadb.Client()
        
        try:
            self._client.delete_collection(self._config.collection_name)
        except Exception:
            pass
            
        self._collection = self._client.create_collection(
            name=self._config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Cache for full content to avoid redundant downloads
        cache_dir = Path(".cache/full_content")
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"      [RAG] Inflating {len(self._pages)} pages with full content...")
        
        url_to_slugs = {} # URL -> list of slugs
        slug_to_page = {p.slug: p for p in self._pages}
        all_ids = []

        # 1. Map URLs to slugs and handle summaries
        pages_to_process = []
        seen_urls = set()

        for page in self._pages:
            url_match = re.search(r"^ref: (.+)$", page.content, re.MULTILINE)
            if not url_match:
                # No URL, just index the summary as a unique "page"
                pages_to_process.append((page.slug, page.content, page.title))
            else:
                url = url_match.group(1).strip()
                if url not in url_to_slugs:
                    url_to_slugs[url] = []
                    pages_to_process.append((url, None, page.title, page.slug)) # url, content(fetch later), title, sample_slug
                url_to_slugs[url].append(page.slug)

        # 2. Fetch and chunk
        headers = {
            "User-Agent": "SemaTree/0.1"
        }
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=headers) as client:
            for item in pages_to_process:
                if len(item) == 3: # Summary only
                    slug, content, title = item
                    url = None
                else: # Full content fetch
                    url, _, title, sample_slug = item
                    cache_path = cache_dir / f"{sample_slug}.md"
                    
                    if cache_path.exists():
                        content = cache_path.read_text(encoding="utf-8")
                    else:
                        await asyncio.sleep(0.5) # Be kind to servers
                        try:
                            resp = await client.get(url)
                            resp.raise_for_status()
                            soup = BeautifulSoup(resp.text, "html.parser")
                            for tag in soup(["script", "style", "nav", "footer", "header"]):
                                tag.decompose()
                            main = soup.find("main") or soup.find("article") or soup.body or soup
                            content = markdownify(str(main), heading_style="ATX").strip()
                            cache_path.write_text(content, encoding="utf-8")
                        except Exception as e:
                            logger.error("Failed to fetch full content for %s: %s. Falling back to index summary.", url, e)
                            content = slug_to_page[sample_slug].content

                chunks = _chunk_text(content, self._config.chunk_size, self._config.chunk_overlap)
                for i, chunk in enumerate(chunks):
                    idx = len(self._all_chunks)
                    self._all_chunks.append(chunk)
                    all_ids.append(f"chunk_{idx}")
                    
                    # Store all original slugs for this content
                    source_slugs = url_to_slugs[url] if url else [slug]
                    self._all_metadatas.append({
                        "sources": json.dumps(source_slugs), 
                        "title": title, 
                        "index": idx
                    })

        # 1. Setup Vector Search
        logger.info("Embedding %d chunks...", len(self._all_chunks))
        embeddings = self._embedder.encode(self._all_chunks, show_progress_bar=False)
        self._collection.add(
            ids=all_ids,
            documents=self._all_chunks,
            embeddings=[emb.tolist() for emb in embeddings],
            metadatas=[{"sources": m["sources"], "index": m["index"]} for m in self._all_metadatas],
        )

        # 2. Setup BM25
        tokenized_corpus = [doc.lower().split() for doc in self._all_chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Hybrid index built (Vector + BM25)")

    async def retrieve(self, question: str) -> RetrievalResult:
        """Perform Hybrid Search using Reciprocal Rank Fusion."""
        assert self._collection is not None and self._bm25 is not None
        t0 = time.perf_counter()

        # 1. Vector Search (Semantic)
        query_embedding = self._embedder.encode([question])[0].tolist()
        vector_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=self._config.top_k * 2,
            include=["documents", "metadatas"],
        )
        
        # 2. BM25 Search (Keyword)
        tokenized_query = question.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:self._config.top_k * 2]

        # 3. Reciprocal Rank Fusion (RRF)
        k_rrf = 60
        scores = {}  # chunk_index -> rrf_score

        if vector_results["metadatas"]:
            for rank, meta in enumerate(vector_results["metadatas"][0], start=1):
                idx = meta["index"]
                scores[idx] = scores.get(idx, 0) + (1 / (k_rrf + rank))

        for rank, idx in enumerate(top_bm25_indices, start=1):
            scores[idx] = scores.get(idx, 0) + (1 / (k_rrf + rank))

        sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:self._config.top_k]
        
        retrieved_chunks = [self._all_chunks[i] for i in sorted_indices]
        
        # Collect all source slugs correctly
        retrieved_sources = set()
        for i in sorted_indices:
            slugs = json.loads(self._all_metadatas[i]["sources"])
            for s in slugs:
                retrieved_sources.add(s)
        
        context = "\n\n---\n\n".join(retrieved_chunks)

        # 4. Generate answer
        prompt = (
            f"Answer the following question using ONLY the provided context. "
            f"If the context does not contain enough information, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
        gen_resp = await self._provider.generate(user_message=prompt, max_tokens=1024)
        
        latency_ms = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            retrieved_sources=list(retrieved_sources),
            retrieved_content=context,
            answer=gen_resp.text,
            tokens_used=gen_resp.input_tokens + gen_resp.output_tokens,
            latency_ms=latency_ms,
            api_calls=1,
            input_tokens=gen_resp.input_tokens,
            output_tokens=gen_resp.output_tokens,
            embedding_tokens=0,
            model=self._config.generation_model,
        )

    async def teardown(self) -> None:
        if self._client is not None:
            try:
                self._client.delete_collection(self._config.collection_name)
            except Exception:
                pass
        self._embedder = None
