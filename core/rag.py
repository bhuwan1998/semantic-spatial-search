"""
Hybrid RAG Module - pgvector-based schema retrieval for context-efficient prompting.

Instead of injecting the full schema (~3,000 tokens) into every LLM call,
this module embeds the user query and performs a cosine similarity search
against the table_descriptions table to retrieve only the most relevant
table schemas (~4 tables, ~800 tokens).

Implements the Semantic-Spatial Fusion approach from GeoAgentic-RAG (Liang et al., 2026).

Graceful degradation: if the embedding model or pgvector is unavailable,
retrieve() returns None and the caller falls back to full schema injection.
"""

import os

import ollama

from core.db import get_conn


class HybridRAG:
    """Semantic schema retrieval via pgvector cosine similarity."""

    def __init__(self, embedding_model: str | None = None, top_k: int = 4):
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.top_k = top_k

    def _embed(self, text: str) -> list[float] | None:
        """Embed text using Ollama. Returns None on failure."""
        try:
            response = ollama.embed(model=self.embedding_model, input=text)
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
