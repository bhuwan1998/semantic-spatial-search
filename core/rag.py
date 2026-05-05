"""
Hybrid RAG Module - pgvector-based schema retrieval for context-efficient prompting.

Instead of injecting the full schema (~3,000 tokens) into every LLM call,
this module embeds the user query and performs a cosine similarity search
against the table_descriptions table to retrieve only the most relevant
table schemas (~4 tables, ~800 tokens).

Ideal RAG pipeline
------------------
Embedding + retrieval should happen BEFORE the user submits a query so the
result is ready the moment the LLM call starts.  This module supports that via:

  1. retrieve_cached(query)   — LRU-cached version of retrieve(); if the same
                                query (or one already pre-fetched) is requested
                                again the embed + pgvector call is skipped entirely.

  2. prefetch(queries)        — fires retrieve_cached() for a list of queries in
                                a background ThreadPoolExecutor so the cache is
                                warm before the user interacts.  Call once on
                                app startup with the example query list.

  3. prefetch_async(query)    — fire-and-forget single query prefetch; call from
                                an on_change handler or whenever the user starts
                                typing to pre-warm the cache for the likely query.

Graceful degradation: if the embedding model or pgvector is unavailable,
retrieve() returns None and the caller falls back to full schema injection.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import ollama

from core.db import get_conn

# Module-level singleton RAG instance and executor shared across all calls.
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag_prefetch")


class HybridRAG:
    """Semantic schema retrieval via pgvector cosine similarity."""

    def __init__(self, embedding_model: str | None = None, top_k: int = 4):
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.top_k = top_k

    def _embed(self, text: str) -> list[float] | None:
        """Embed text using Ollama. Returns None on failure."""
        try:
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            client = ollama.Client(host=ollama_host)
            response = client.embed(model=self.embedding_model, input=text)
            return response["embeddings"][0]
        except Exception:
            return None

    def retrieve(self, query: str) -> list[str] | None:
        """
        Return a list of the top-k most relevant table names for the query.
        Returns None if embedding or DB lookup fails (caller uses full schema).
        """
        vec = self._embed(query)
        if vec is None:
            return None

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT table_name
                        FROM table_descriptions
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (vec, self.top_k),
                    )
                    rows = cur.fetchall()
                    return [r["table_name"] for r in rows]
        except Exception:
            return None

    def retrieve_with_descriptions(self, query: str) -> list[dict] | None:
        """
        Return top-k rows with table_name and description.
        Used for diagnostic/debug purposes.
        """
        vec = self._embed(query)
        if vec is None:
            return None

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT table_name, description,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM table_descriptions
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (vec, vec, self.top_k),
                    )
                    return cur.fetchall()
        except Exception:
            return None


# ─── module-level cached retrieve ─────────────────────────────────────────────
# Keyed on (query, embedding_model, top_k) so different model configs stay
# independent.  maxsize=512 covers a full session of unique queries.

@lru_cache(maxsize=512)
def _retrieve_cached_impl(query: str, embedding_model: str, top_k: int) -> tuple[str, ...] | None:
    """
    Internal cached implementation.  Returns a tuple (hashable for lru_cache)
    or None on failure.  Do not call directly — use retrieve_cached().
    """
    rag = HybridRAG(embedding_model=embedding_model, top_k=top_k)
    result = rag.retrieve(query)
    return tuple(result) if result is not None else None


def retrieve_cached(
    query: str,
    embedding_model: str | None = None,
    top_k: int = 4,
) -> list[str] | None:
    """
    LRU-cached version of HybridRAG.retrieve().

    First call for a given query pays the full embed + pgvector cost (~1.3s).
    Subsequent calls with the same query string return instantly from the
    in-process cache — no Ollama round-trip, no DB query.

    Returns list[str] of table names, or None on failure.
    """
    model = embedding_model or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    result = _retrieve_cached_impl(query, model, top_k)
    return list(result) if result is not None else None


def prefetch(queries: list[str], embedding_model: str | None = None, top_k: int = 4) -> None:
    """
    Warm the LRU cache for a list of queries in a background thread pool.

    Call once on app startup with the EXAMPLE_QUERIES list.  By the time a
    user clicks an example query chip the result is already cached — zero wait.

    Non-blocking: returns immediately, cache fills in the background.
    """
    model = embedding_model or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    def _warm(q: str) -> None:
        try:
            _retrieve_cached_impl(q, model, top_k)
        except Exception:
            pass

    for q in queries:
        _executor.submit(_warm, q)


def prefetch_async(query: str, embedding_model: str | None = None, top_k: int = 4) -> None:
    """
    Fire-and-forget prefetch for a single query.

    Call from an on_change handler or whenever the input widget changes so
    the embed is in-flight while the user finishes typing/thinking.
    Non-blocking.
    """
    model = embedding_model or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    def _warm() -> None:
        try:
            _retrieve_cached_impl(query, model, top_k)
        except Exception:
            pass

    _executor.submit(_warm)
