"""
indexer/chunker.py
──────────────────
Lexer + Chunker pipeline.

Supported file types:
  .py   → token-aware chunking via Python's `tokenize` module
          (splits on class/function boundaries)
  .md   → splits on heading boundaries (## / ###)
  .pdf  → page-level extraction via pdfminer.six, then heading split
  *     → plain-text fallback with sliding-window word chunks

Each chunk is a dict:
  {
    "id":        str,   # "<filename>:<chunk_index>"
    "source":    str,   # original file path
    "file_type": str,   # "python" | "markdown" | "pdf" | "text"
    "heading":   str,   # nearest heading / function name (may be "")
    "text":      str,   # the actual content
    "char_count":int,
  }
"""

from __future__ import annotations

import hashlib
import io
import re
import tokenize
from pathlib import Path
from typing import Generator


# ── helpers ──────────────────────────────────────────────────────────────────

def _chunk_id(source: str, idx: int) -> str:
    short = Path(source).name
    return f"{short}:{idx}"


def _word_window(text: str, window: int = 400, overlap: int = 80) -> list[str]:
    """Sliding-window fallback for plain text."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + window]))
        i += window - overlap
    return chunks


# ── Python lexer ─────────────────────────────────────────────────────────────

def _split_python(source: str, text: str) -> list[dict]:
    """
    Split a Python file on top-level class / function definitions.
    Each definition becomes its own chunk. Code between definitions
    (imports, module-level statements) forms a preamble chunk.
    """
    lines = text.splitlines(keepends=True)
    boundaries: list[tuple[int, str]] = []  # (line_no_0indexed, heading)

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except tokenize.TokenError:
        tokens = []

    for tok in tokens:
        if tok.type == tokenize.NAME and tok.string in ("class", "def"):
            line_no = tok.start[0] - 1  # 0-indexed
            # grab the next NAME token as the identifier
            idx = tokens.index(tok)
            for nxt in tokens[idx + 1 :]:
                if nxt.type == tokenize.NAME:
                    boundaries.append((line_no, f"{tok.string} {nxt.string}"))
                    break

    if not boundaries:
        # no class/def found — treat whole file as one chunk
        return [
            {
                "id": _chunk_id(source, 0),
                "source": source,
                "file_type": "python",
                "heading": Path(source).name,
                "text": text,
                "char_count": len(text),
            }
        ]

    # build slices: preamble + each definition until next boundary
    slices: list[tuple[str, str]] = []
    if boundaries[0][0] > 0:
        slices.append(("module preamble", "".join(lines[: boundaries[0][0]])))

    for i, (start, heading) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
        slices.append((heading, "".join(lines[start:end])))

    chunks = []
    for idx, (heading, chunk_text) in enumerate(slices):
        if chunk_text.strip():
            chunks.append(
                {
                    "id": _chunk_id(source, idx),
                    "source": source,
                    "file_type": "python",
                    "heading": heading,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                }
            )
    return chunks


# ── Markdown lexer ───────────────────────────────────────────────────────────

def _split_markdown(source: str, text: str) -> list[dict]:
    """Split on ## and ### headings."""
    pattern = re.compile(r"^(#{1,3} .+)$", re.MULTILINE)
    positions = [(m.start(), m.group(1)) for m in pattern.finditer(text)]

    if not positions:
        return [
            {
                "id": _chunk_id(source, 0),
                "source": source,
                "file_type": "markdown",
                "heading": Path(source).stem,
                "text": text,
                "char_count": len(text),
            }
        ]

    chunks = []
    # content before first heading
    if positions[0][0] > 0:
        preamble = text[: positions[0][0]].strip()
        if preamble:
            chunks.append(
                {
                    "id": _chunk_id(source, 0),
                    "source": source,
                    "file_type": "markdown",
                    "heading": "preamble",
                    "text": preamble,
                    "char_count": len(preamble),
                }
            )

    for i, (pos, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        chunk_text = text[pos:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "id": _chunk_id(source, i + 1),
                    "source": source,
                    "file_type": "markdown",
                    "heading": heading.lstrip("#").strip(),
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                }
            )
    return chunks


# ── PDF lexer ────────────────────────────────────────────────────────────────

def _split_pdf(source: str) -> list[dict]:
    """
    Extract text page-by-page via pdfminer.six, then apply markdown
    heading split on each page's text.
    Falls back to plain chunking if pdfminer is not installed.
    """
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer
    except ImportError:
        # graceful fallback: read raw bytes, try utf-8
        raw = Path(source).read_bytes()
        text = raw.decode("utf-8", errors="replace")
        return _split_plain(source, text, file_type="pdf")

    pages_text: list[str] = []
    for page_layout in extract_pages(source):
        page_lines = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_lines.append(element.get_text())
        pages_text.append("".join(page_lines))

    all_chunks: list[dict] = []
    for page_no, page_text in enumerate(pages_text):
        sub = _split_markdown(source, page_text)
        for i, chunk in enumerate(sub):
            chunk["id"] = _chunk_id(source, page_no * 100 + i)
            chunk["file_type"] = "pdf"
            chunk["heading"] = f"page {page_no + 1} — {chunk['heading']}"
            all_chunks.append(chunk)

    return all_chunks or [
        {
            "id": _chunk_id(source, 0),
            "source": source,
            "file_type": "pdf",
            "heading": Path(source).stem,
            "text": "\n".join(pages_text),
            "char_count": sum(len(p) for p in pages_text),
        }
    ]


# ── plain-text fallback ───────────────────────────────────────────────────────

def _split_plain(source: str, text: str, file_type: str = "text") -> list[dict]:
    windows = _word_window(text)
    return [
        {
            "id": _chunk_id(source, i),
            "source": source,
            "file_type": file_type,
            "heading": f"chunk {i}",
            "text": w,
            "char_count": len(w),
        }
        for i, w in enumerate(windows)
        if w.strip()
    ]


# ── public API ────────────────────────────────────────────────────────────────

def chunk_file(path: str | Path) -> list[dict]:
    """
    Main entry point.  Accepts a file path, returns a list of chunk dicts.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = p.suffix.lower()

    if suffix == ".pdf":
        return _split_pdf(str(p))

    text = p.read_text(encoding="utf-8", errors="replace")

    if suffix == ".py":
        return _split_python(str(p), text)
    elif suffix in (".md", ".mdx", ".rst"):
        return _split_markdown(str(p), text)
    else:
        return _split_plain(str(p), text)


def chunk_directory(
    directory: str | Path,
    extensions: tuple[str, ...] = (".py", ".md", ".pdf", ".txt", ".rst"),
    recursive: bool = True,
) -> Generator[dict, None, None]:
    """
    Walk a directory and yield chunks from every matching file.
    """
    d = Path(directory)
    glob = d.rglob("*") if recursive else d.glob("*")
    for file_path in sorted(glob):
        if file_path.suffix.lower() in extensions and file_path.is_file():
            try:
                for chunk in chunk_file(file_path):
                    yield chunk
            except Exception as exc:
                print(f"[chunker] skipping {file_path}: {exc}")