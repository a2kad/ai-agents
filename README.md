# AI Agents — Local Multi-Agent Code Pipeline

This repository provides a local multi-agent pipeline for code generation and review using Ollama models. It is optimized for Apple Silicon (M1/M2/M3) and designed so all data and models remain on your machine.

Quick highlights:

- Local-first: models run via Ollama on your computer.
- RAG-enabled: project indexing and retrieval for context-aware generation.
- Extensible: add agents and swap models easily.

## Quickstart

1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Pull required Ollama models (once):

```bash
ollama pull llama3.2:3b
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

4. Run the interactive pipeline:

```bash
source venv/bin/activate
python pipeline.py
```

## Overview

The pipeline routes user tasks to specialized agents:

- Router agent (`llama3.2:3b`) — classifies the task.
- Code generator (`qwen2.5-coder:7b`) — generates code for `CODE` tasks.
- Reviewer (`qwen2.5:14b`) — performs reviews and complex analysis.
- `nomic-embed-text` — embeddings for RAG (indexing & search).

RAG (Retrieval-Augmented Generation) lets agents index your repository so generated code respects your project's style and conventions.

## Indexing (RAG)

To index a directory into the local Qdrant store:

```bash
./venv/bin/python -m indexer.index_docs ./path/to/project
```

Use `--reset` to drop and recreate the collection.

## Development

- Run unit tests:

```bash
pytest tests/ -q
```

- Linting and formatting: use your preferred tools (Black, Flake8, etc.).

## Contributing

1. Fork the repo and create a feature branch.
2. Add tests for new behavior.
3. Open a PR with a clear description.

## Files of interest

- `pipeline.py` — main pipeline entrypoint.
- `agents.py` — agent implementations.
- `indexer/` — chunking and embedding helpers.
- `rag/` — vector store wrapper and RAG helpers.

## Troubleshooting

- If you see Qdrant shutdown warnings during CLI exit, update to the latest `qdrant-client` and ensure the client is closed explicitly (the CLI already calls `close_client()` on exit).
- If Ollama models fail to load, confirm `ollama` is running and models are pulled.

## License

MIT