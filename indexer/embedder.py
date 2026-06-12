"""
indexer/embedder.py
────────────────────
Converts text chunks → float vectors using Ollama's nomic-embed-text.

Batches requests to avoid overwhelming the local model server.
Includes a simple retry with exponential back-off.
"""

from __future__ import annotations

import time
from typing import Sequence

import ollama

EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 16          # chunks per Ollama call
MAX_RETRIES = 3
BACKOFF_BASE = 1.5       # seconds


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Call Ollama embeddings for a list of texts (one API call each — Ollama
    does not support batch embed natively, so we call in a tight loop but
    keep the retry logic centralised here)."""
    vectors: list[list[float]] = []
    for text in texts:
        for attempt in range(MAX_RETRIES):
            try:
                resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
                vectors.append(resp["embedding"])
                break
            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"Embedding failed after {MAX_RETRIES} attempts: {exc}"
                    ) from exc
                wait = BACKOFF_BASE ** attempt
                print(f"[embedder] retry {attempt+1} in {wait:.1f}s — {exc}")
                time.sleep(wait)
    return vectors


def embed_chunks(chunks: Sequence[dict]) -> list[dict]:
    """
    Takes a list of chunk dicts (from chunker.py) and returns the same list
    with an added  "vector": list[float]  field on each chunk.

    Processes in batches of BATCH_SIZE for progress visibility.
    """
    enriched: list[dict] = []
    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):
        batch = list(chunks[start : start + BATCH_SIZE])
        texts = [c["text"] for c in batch]

        print(
            f"[embedder] embedding chunks {start+1}–{min(start+BATCH_SIZE, total)}"
            f" / {total}"
        )
        vectors = _embed_batch(texts)

        for chunk, vector in zip(batch, vectors):
            enriched.append({**chunk, "vector": vector})

    return enriched


def embed_query(query: str) -> list[float]:
    """Embed a single query string for similarity search."""
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=query)
    return resp["embedding"]