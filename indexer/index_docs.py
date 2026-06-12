"""
indexer/index_docs.py
──────────────────────
CLI tool: chunk → embed → upsert into Qdrant.

Usage:
    python -m indexer.index_docs ./my_project
    python -m indexer.index_docs ./docs --ext .md .rst
    python -m indexer.index_docs ./src  --reset
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# make sure package root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer.chunker import chunk_directory
from indexer.embedder import embed_chunks
from rag.vector_store import upsert_chunks, delete_collection, collection_info, close_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Index documents into Qdrant")
    parser.add_argument("directory", help="Path to directory to index")
    parser.add_argument(
        "--ext",
        nargs="+",
        default=[".py", ".md", ".pdf", ".txt", ".rst"],
        help="File extensions to include (default: .py .md .pdf .txt .rst)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the collection before indexing",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recurse into subdirectories",
    )
    args = parser.parse_args()

    try:
        directory = Path(args.directory)
        if not directory.exists():
            print(f"[index_docs] ERROR: directory not found: {directory}")
            sys.exit(1)

        if args.reset:
            print("[index_docs] resetting collection …")
            delete_collection()

        extensions = tuple(
            e if e.startswith(".") else f".{e}" for e in args.ext
        )
        recursive = not args.no_recursive

        print(f"[index_docs] scanning {directory} (extensions: {extensions}) …")
        t0 = time.time()

        # ── 1. Chunk ──────────────────────────────────────────────────────────────
        all_chunks = list(
            chunk_directory(directory, extensions=extensions, recursive=recursive)
        )
        print(f"[index_docs] {len(all_chunks)} chunks from {directory}")

        if not all_chunks:
            print("[index_docs] nothing to index. Exiting.")
            return

        # ── 2. Embed ──────────────────────────────────────────────────────────────
        embedded = embed_chunks(all_chunks)

        # ── 3. Upsert ─────────────────────────────────────────────────────────────
        n = upsert_chunks(embedded)

        elapsed = time.time() - t0
        info = collection_info()

        print(
            f"\n[index_docs] ✅ done in {elapsed:.1f}s\n"
            f"  upserted : {n} chunks\n"
            f"  total in collection : {info['points_count']}\n"
            f"  collection status   : {info['status']}"
        )
    finally:
        close_client()


if __name__ == "__main__":
    main()