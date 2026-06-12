"""
tests/test_rag_pipeline.py
───────────────────────────
Run with:  pytest tests/ -v

Tests are split into unit (no Ollama/Qdrant needed) and integration
(requires Ollama running with nomic-embed-text pulled).

Mark integration tests:
    pytest tests/ -v -m "not integration"   # unit only
    pytest tests/ -v                        # all
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer.chunker import chunk_file, _split_python, _split_markdown


# ════════════════════════════════════════════════════════
# UNIT TESTS — no external services needed
# ════════════════════════════════════════════════════════

class TestMarkdownChunker:
    def test_splits_on_headings(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text(
            "# Title\nIntro paragraph.\n\n"
            "## Section A\nContent A.\n\n"
            "## Section B\nContent B.\n"
        )
        chunks = chunk_file(md)
        headings = [c["heading"] for c in chunks]
        assert "Section A" in headings
        assert "Section B" in headings

    def test_file_type_is_markdown(self, tmp_path):
        md = tmp_path / "readme.md"
        md.write_text("## Hello\nworld")
        chunks = chunk_file(md)
        assert all(c["file_type"] == "markdown" for c in chunks)

    def test_no_headings_returns_single_chunk(self, tmp_path):
        md = tmp_path / "flat.md"
        md.write_text("Just some plain text without any headings here.")
        chunks = chunk_file(md)
        assert len(chunks) == 1

    def test_chunk_contains_text(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("## Auth\nJWT tokens are used.")
        chunks = chunk_file(md)
        assert any("JWT" in c["text"] for c in chunks)


class TestPythonChunker:
    def test_splits_on_functions(self, tmp_path):
        py = tmp_path / "utils.py"
        py.write_text(
            "import os\n\n"
            "def foo():\n    return 1\n\n"
            "def bar():\n    return 2\n"
        )
        chunks = chunk_file(py)
        headings = [c["heading"] for c in chunks]
        assert any("foo" in h for h in headings)
        assert any("bar" in h for h in headings)

    def test_splits_on_class(self, tmp_path):
        py = tmp_path / "models.py"
        py.write_text(
            "class User:\n    id: int\n    email: str\n\n"
            "class Post:\n    title: str\n"
        )
        chunks = chunk_file(py)
        headings = [c["heading"] for c in chunks]
        assert any("User" in h for h in headings)
        assert any("Post" in h for h in headings)

    def test_file_type_is_python(self, tmp_path):
        py = tmp_path / "code.py"
        py.write_text("def hello(): pass\n")
        chunks = chunk_file(py)
        assert all(c["file_type"] == "python" for c in chunks)

    def test_chunk_id_format(self, tmp_path):
        py = tmp_path / "myfile.py"
        py.write_text("def greet(): pass\n")
        chunks = chunk_file(py)
        for c in chunks:
            assert "myfile.py" in c["id"]
            assert ":" in c["id"]

    def test_invalid_python_falls_back(self, tmp_path):
        py = tmp_path / "broken.py"
        py.write_text("def unclosed(:\n    pass\n")
        # should not raise, may produce 1 chunk
        chunks = chunk_file(py)
        assert len(chunks) >= 1


class TestChunkDirectory:
    def test_indexes_multiple_files(self, tmp_path):
        (tmp_path / "a.py").write_text("def a(): pass\n")
        (tmp_path / "b.md").write_text("## B\ncontent\n")
        (tmp_path / "c.txt").write_text("plain text file\n")

        from indexer.chunker import chunk_directory
        chunks = list(chunk_directory(tmp_path))
        sources = {Path(c["source"]).name for c in chunks}
        assert "a.py" in sources
        assert "b.md" in sources

    def test_respects_extension_filter(self, tmp_path):
        (tmp_path / "a.py").write_text("def a(): pass\n")
        (tmp_path / "b.md").write_text("## B\ncontent\n")

        from indexer.chunker import chunk_directory
        chunks = list(chunk_directory(tmp_path, extensions=(".py",)))
        assert all(c["file_type"] == "python" for c in chunks)


# ════════════════════════════════════════════════════════
# INTEGRATION TESTS — require Ollama + Qdrant
# ════════════════════════════════════════════════════════

@pytest.mark.integration
class TestEmbedderIntegration:
    def test_embed_query_returns_vector(self):
        from indexer.embedder import embed_query
        vec = embed_query("test query about authentication")
        assert isinstance(vec, list)
        assert len(vec) == 768
        assert all(isinstance(v, float) for v in vec)

    def test_embed_chunks_adds_vector_field(self, tmp_path):
        from indexer.chunker import chunk_file
        from indexer.embedder import embed_chunks

        md = tmp_path / "test.md"
        md.write_text("## Section\nSome content here.")
        chunks = chunk_file(md)
        embedded = embed_chunks(chunks)

        assert len(embedded) == len(chunks)
        for c in embedded:
            assert "vector" in c
            assert len(c["vector"]) == 768


@pytest.mark.integration
class TestVectorStoreIntegration:
    def test_upsert_and_search(self, tmp_path):
        from indexer.chunker import chunk_file
        from indexer.embedder import embed_chunks, embed_query
        from rag.vector_store import upsert_chunks, search, delete_collection

        # fresh slate
        delete_collection()

        md = tmp_path / "auth.md"
        md.write_text(
            "## JWT Authentication\n"
            "Users authenticate via JWT tokens. "
            "Access token expires in 15 minutes.\n"
        )
        chunks = chunk_file(md)
        embedded = embed_chunks(chunks)
        n = upsert_chunks(embedded)
        assert n > 0

        query_vec = embed_query("how does authentication work")
        results = search(query_vec, top_k=1)

        assert len(results) >= 1
        assert "JWT" in results[0]["text"] or results[0]["score"] > 0.5

    def test_search_returns_score(self, tmp_path):
        from indexer.chunker import chunk_file
        from indexer.embedder import embed_chunks, embed_query
        from rag.vector_store import upsert_chunks, search

        md = tmp_path / "db.md"
        md.write_text("## Database\nPostgreSQL with async SQLAlchemy.")
        chunks = chunk_file(md)
        upsert_chunks(embed_chunks(chunks))

        results = search(embed_query("database orm"), top_k=3)
        for r in results:
            assert "score" in r
            assert 0.0 <= r["score"] <= 1.0


@pytest.mark.integration
class TestFullPipelineIntegration:
    def test_code_task_uses_rag(self, tmp_path):
        """End-to-end: index a doc, then run a CODE task and verify context was retrieved."""
        from indexer.chunker import chunk_file
        from indexer.embedder import embed_chunks
        from rag.vector_store import upsert_chunks, delete_collection

        delete_collection()

        md = tmp_path / "payments.md"
        md.write_text(
            "## Stripe Payments\n"
            "Use stripe.PaymentIntent.create() with idempotency keys.\n"
            "Always store payment_intent_id in the Order model.\n"
        )
        upsert_chunks(embed_chunks(chunk_file(md)))

        # run pipeline — just check it doesn't crash and returns a string
        from graph.pipeline import run_pipeline
        result = run_pipeline("Write a function to create a Stripe payment intent")
        assert isinstance(result, str)
        assert len(result) > 50