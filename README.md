# 🤖 AI Code Pipeline — Apple M3 Pro

A local multi-agent code development pipeline powered by Ollama. All models run on your MacBook — your data never leaves the machine.

---

## Architecture

```
User task
        │
        ▼
┌───────────────────┐
│   Router agent    │  llama3.2:3b  — classifies the task
└────────┬──────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
SIMPLE      CODE ──────────────────────┐
  │           │                        │
  ▼           ▼                        ▼
llama3.2   qwen2.5-coder:7b      REVIEW / COMPLEX
  :3b      code generation        qwen2.5:14b
                │                 analysis & review
                ▼
         Auto-review
         qwen2.5:14b
```

---

## Models

| Model | Role | VRAM / RAM |
|---|---|---|
| `llama3.2:3b` | Router, classifier, simple questions | ~2 GB |
| `qwen2.5-coder:7b` | Code generation | ~4.7 GB |
| `qwen2.5:14b` | Review, architecture, complex analysis | ~8.9 GB |
| `nomic-embed-text` | Embeddings for RAG | ~0.3 GB |

---

## Requirements

- Apple M3 Pro, 18 GB RAM
- macOS 13+
- Python 3.11+
- [Ollama](https://ollama.com) installed and running

---

## Installation

### 1. Ollama and models

```bash
# Install Ollama from ollama.com, then:
ollama pull llama3.2:3b
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

### 2. Python environment

```bash
git clone https://github.com/a2kad/ai-agents.git
cd ai-agents

python3 -m venv venv
source venv/bin/activate

pip install ollama langgraph langchain-ollama chromadb rich
```

---

## Project structure

```
ai-agents/
├── venv/
├── test_connection.py   # Verify connection to all models
├── router_agent.py      # Task classifier (llama3.2:3b)
├── agents.py            # Specialized agents
├── pipeline.py          # Main pipeline — entry point
└── rag_agent.py         # RAG agent with project memory
```

---

## Usage

### Interactive mode

```bash
source venv/bin/activate
python pipeline.py
```

Type your task in any language — the pipeline will route it to the right agent automatically.

### Example prompts

```
Your task: Write a Python class for a queue data structure
→ Router: CODE → qwen2.5-coder:7b → auto-review by qwen2.5:14b

Your task: Find bugs: def divide(a, b): return a / b
→ Router: REVIEW → qwen2.5:14b

Your task: What is a Python decorator?
→ Router: SIMPLE → llama3.2:3b

Your task: Design a microservices architecture for an e-commerce platform
→ Router: COMPLEX → qwen2.5:14b
```

### Health check

```bash
python test_connection.py
# Expected output:
# llama3.2:3b: ok
# qwen2.5-coder:7b: ok
# qwen2.5:14b: ok
```

---

## RAG — project memory

Indexes your codebase so agents understand your project and generate code that matches your style and conventions.

```bash
python rag_agent.py
```

Add documents manually:

```python
from rag_agent import add_document

add_document("auth", "JWT authentication, User model: id, email, role")
add_document("db",   "PostgreSQL, SQLAlchemy ORM, Alembic migrations")
add_document("api",  "FastAPI, all endpoints return {data, error, meta}")
```

Once indexed, agents will respect your conventions when generating code.

---

## Git Hook — auto-review before commit

Set up once, runs automatically on every `git commit`.

```bash
# From the root of your project:
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
cd /path/to/ai-agents && source venv/bin/activate
git diff --cached --name-only --diff-filter=M | grep ".py$" | while read file; do
    python -c "
from agents import review_agent
import sys
code = open('$file').read()
result = review_agent(f'Review this code:\n{code}')
print(f'[REVIEW] $file\n{result}')
if 'score: [1-4]' in result.lower():
    sys.exit(1)
"
done
EOF
chmod +x .git/hooks/pre-commit
```

---

## Performance on M3 Pro

| Model | Speed | Typical response time |
|---|---|---|
| `llama3.2:3b` (router) | ~70 tok/s | < 1 sec |
| `qwen2.5-coder:7b` | ~35 tok/s | 5–15 sec |
| `qwen2.5:14b` (review) | ~18 tok/s | 15–40 sec |

The M3 Pro unified memory allows all models to stay loaded simultaneously with no reloading between requests.

---

## Extending the pipeline

### Add a new agent

```python
# in agents.py
def test_agent(task: str) -> str:
    response = ollama.chat(
        model="qwen2.5-coder:7b",
        messages=[{
            "role": "system",
            "content": "You are a testing expert. Write pytest tests including edge cases."
        }, {
            "role": "user", "content": task
        }]
    )
    return response['message']['content']
```

```python
# in pipeline.py — add to AGENT_MAP:
"TEST": (test_agent, "qwen2.5-coder:7b", "blue"),
```

### Swap in a different model

```bash
ollama pull deepseek-r1:14b   # strong reasoning
ollama pull phi4:14b          # compact and fast
```

Replace the `model=` value in the relevant agent inside `agents.py`.

---

## VS Code integration

Install the [Continue](https://continue.dev) extension and point it to your local Ollama:

```json
{
  "models": [
    {
      "title": "Qwen Coder",
      "provider": "ollama",
      "model": "qwen2.5-coder:7b"
    }
  ]
}
```

The agent will then be available directly in the editor via `Cmd+I`.

---

## License

MIT — free to use.

---

*Built with: Ollama · Qwen2.5 · Llama 3.2 · LangGraph · ChromaDB*