"""
rag/vector_store.py
────────────────────
Qdrant client wrapper.

Uses Qdrant in **local / in-process** mode (no Docker required) by default:
data is persisted to  ./qdrant_data  next to this file.

To switch to a running Qdrant server, set env var:
    QDRANT_URL=http://localhost:6333

Collection schema
─────────────────
  vector size : 768   (nomic-embed-text output dimension)
  distance    : Cosine
  payload keys: id, source, file_type, heading, text, char_count
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

COLLECTION_NAME = "codebase"
VECTOR_SIZE = 768          # nomic-embed-text
TOP_K_DEFAULT = 3
LOCAL_PATH = str(Path(__file__).parent.parent / "qdrant_data")


def _make_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "")
    if url:
        print(f"[vector_store] connecting to Qdrant at {url}")
        return QdrantClient(url=url)
    print(f"[vector_store] using local Qdrant storage at {LOCAL_PATH}")
    return QdrantClient(path=LOCAL_PATH)


_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = _make_client()
        _ensure_collection(_client)
    return _client


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


def _ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"[vector_store] created collection '{COLLECTION_NAME}'")
    else:
        print(f"[vector_store] collection '{COLLECTION_NAME}' already exists")


# ── write ─────────────────────────────────────────────────────────────────────

def upsert_chunks(chunks_with_vectors: Sequence[dict]) -> int:
    """
    Upsert a list of embedded chunks into Qdrant.
    Each chunk must have a 'vector' key (list[float]).
    Returns the number of points upserted.
    """
    client = get_client()
    points: list[PointStruct] = []

    for chunk in chunks_with_vectors:
        vector = chunk.get("vector")
        if not vector:
            print(f"[vector_store] skipping chunk with no vector: {chunk['id']}")
            continue

        # deterministic UUID from chunk id so re-indexing is idempotent
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"]))

        payload = {
            "chunk_id":  chunk["id"],
            "source":    chunk["source"],
            "file_type": chunk["file_type"],
            "heading":   chunk["heading"],
            "text":      chunk["text"],
            "char_count": chunk["char_count"],
        }
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    if not points:
        return 0

    # upsert in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + batch_size],
        )

    print(f"[vector_store] upserted {len(points)} points")
    return len(points)


# ── read ──────────────────────────────────────────────────────────────────────

def search(
    query_vector: list[float],
    top_k: int = TOP_K_DEFAULT,
    file_type_filter: str | None = None,
) -> list[dict]:
    """
    Return top_k most similar chunks.

    Optional file_type_filter: "python" | "markdown" | "pdf" | "text"

    Returns list of dicts:
      { source, file_type, heading, text, score }
    """
    client = get_client()

    qdrant_filter = None
    if file_type_filter:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="file_type",
                    match=MatchValue(value=file_type_filter),
                )
            ]
        )

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    return [
        {
            "source":    r.payload["source"],
            "file_type": r.payload["file_type"],
            "heading":   r.payload["heading"],
            "text":      r.payload["text"],
            "score":     round(r.score, 4),
        }
        for r in results
    ]


# ── maintenance ───────────────────────────────────────────────────────────────

def collection_info() -> dict:
    client = get_client()
    info = client.get_collection(COLLECTION_NAME)
    return {
        "vectors_count": getattr(info, "vectors_count", getattr(info, "indexed_vectors_count", None)),
        "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
        "points_count":  info.points_count,
        "status":        str(info.status),
    }


def delete_collection() -> None:
    """Drop and recreate the collection (full re-index)."""
    client = get_client()
    client.delete_collection(COLLECTION_NAME)
    _ensure_collection(client)
    print("[vector_store] collection reset")