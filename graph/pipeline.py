"""
graph/pipeline.py
──────────────────
LangGraph multi-agent pipeline with RAG.

Graph nodes
───────────
  RouterNode            — classifies the task (llama3.2:3b)
  ContextRetrieverNode  — fetches 2-3 relevant chunks from Qdrant
  CodeAgentNode         — generates code with context (qwen2.5-coder:7b)
  ReviewAgentNode       — reviews code (qwen2.5:14b)
  SimpleAgentNode       — answers simple questions (llama3.2:3b)
  ComplexAgentNode      — architecture / design (qwen2.5:14b)

Routing
───────
  SIMPLE  → SimpleAgentNode  → END
  CODE    → ContextRetrieverNode → CodeAgentNode → ReviewAgentNode → END
  REVIEW  → ReviewAgentNode  → END
  COMPLEX → ContextRetrieverNode → ComplexAgentNode → END
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import ollama
from langgraph.graph import END, StateGraph
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer.embedder import embed_query
from rag.vector_store import search

console = Console()

# ── LangGraph state ───────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    task: str
    category: Literal["SIMPLE", "CODE", "REVIEW", "COMPLEX", ""]
    context: str          # retrieved RAG context
    code_output: str      # generated code
    review_output: str    # review notes
    final_answer: str     # what we show the user


# ── node helpers ──────────────────────────────────────────────────────────────

def _chat(model: str, system: str, user: str) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    resp = ollama.chat(model=model, messages=messages)
    return resp["message"]["content"]


# ── nodes ─────────────────────────────────────────────────────────────────────

def RouterNode(state: PipelineState) -> PipelineState:
    """
    Classify the task into SIMPLE / CODE / REVIEW / COMPLEX.
    Uses llama3.2:3b — fast and cheap.
    """
    prompt = (
        "Classify the task below into exactly one category.\n"
        "Reply with ONE WORD only — no punctuation, no explanation.\n\n"
        "Categories:\n"
        "  SIMPLE  — general question, explanation, documentation lookup\n"
        "  CODE    — write new code, function, class, or feature\n"
        "  REVIEW  — review existing code, find bugs, suggest improvements\n"
        "  COMPLEX — system design, architecture, multi-component planning\n\n"
        f"Task: {state['task']}"
    )
    raw = _chat("llama3.2:3b", "", prompt).strip().upper()
    category: Literal["SIMPLE", "CODE", "REVIEW", "COMPLEX"] = "SIMPLE"
    for cat in ("SIMPLE", "CODE", "REVIEW", "COMPLEX"):
        if cat in raw:
            category = cat  # type: ignore[assignment]
            break

    console.print(f"[bold cyan]🔀 Router → [yellow]{category}[/yellow][/bold cyan]")
    return {**state, "category": category}


def ContextRetrieverNode(state: PipelineState) -> PipelineState:
    """
    Core RAG node.

    1. Embed the task query.
    2. Search Qdrant for top-3 most relevant chunks.
    3. Format them into a context block for the downstream agent.
    """
    console.print("[bold blue]🔍 ContextRetrieverNode — searching Qdrant …[/bold blue]")

    try:
        query_vector = embed_query(state["task"])
        results = search(query_vector, top_k=3)
    except Exception as exc:
        console.print(f"[red]⚠ RAG search failed: {exc}. Proceeding without context.[/red]")
        return {**state, "context": ""}

    if not results:
        console.print("[yellow]  no relevant chunks found[/yellow]")
        return {**state, "context": ""}

    # format context block
    lines = ["### Relevant project context\n"]
    for i, r in enumerate(results, 1):
        source = Path(r["source"]).name
        lines.append(
            f"#### [{i}] {source} — {r['heading']}  (score: {r['score']})\n"
            f"```\n{r['text']}\n```\n"
        )
        console.print(
            f"  [green]✓[/green] {source} / {r['heading']} — score {r['score']}"
        )

    context = "\n".join(lines)
    return {**state, "context": context}


def CodeAgentNode(state: PipelineState) -> PipelineState:
    """Generate code, injecting RAG context into the system prompt."""
    console.print("[bold green]⚙  CodeAgentNode — qwen2.5-coder:7b[/bold green]")

    system = (
        "You are an expert software engineer.\n"
        "Write clean, production-ready code with docstrings and type hints.\n"
        "Follow the conventions shown in the project context below.\n\n"
    )
    if state["context"]:
        system += state["context"]

    code = _chat("qwen2.5-coder:7b", system, state["task"])
    return {**state, "code_output": code, "final_answer": code}


def ReviewAgentNode(state: PipelineState) -> PipelineState:
    """Review code — either from CodeAgentNode output or the raw task."""
    console.print("[bold yellow]🔎 ReviewAgentNode — qwen2.5:14b[/bold yellow]")

    code_to_review = state["code_output"] or state["task"]
    system = (
        "You are a senior code reviewer.\n"
        "Provide a structured review with these sections:\n"
        "1. BUGS — critical issues that must be fixed\n"
        "2. IMPROVEMENTS — non-blocking suggestions\n"
        "3. SCORE — integer 1–10 with one-line rationale\n\n"
    )
    if state["context"]:
        system += "Project conventions for reference:\n" + state["context"]

    review = _chat("qwen2.5:14b", system, f"Review this code:\n{code_to_review}")
    return {**state, "review_output": review, "final_answer": review}


def SimpleAgentNode(state: PipelineState) -> PipelineState:
    """Answer simple / factual questions."""
    console.print("[bold cyan]💬 SimpleAgentNode — llama3.2:3b[/bold cyan]")
    answer = _chat("llama3.2:3b", "You are a helpful assistant.", state["task"])
    return {**state, "final_answer": answer}


def ComplexAgentNode(state: PipelineState) -> PipelineState:
    """Handle architecture / system-design tasks."""
    console.print("[bold red]🏛  ComplexAgentNode — qwen2.5:14b[/bold red]")

    system = (
        "You are an experienced software architect.\n"
        "Provide detailed, structured answers with diagrams in ASCII where helpful.\n\n"
    )
    if state["context"]:
        system += "Existing project context:\n" + state["context"]

    answer = _chat("qwen2.5:14b", system, state["task"])
    return {**state, "final_answer": answer}


# ── routing logic ─────────────────────────────────────────────────────────────

def route_after_router(state: PipelineState) -> str:
    cat = state["category"]
    if cat == "CODE":
        return "retrieve_then_code"
    if cat == "REVIEW":
        return "review"
    if cat == "COMPLEX":
        return "retrieve_then_complex"
    return "simple"


def route_after_code(state: PipelineState) -> str:
    return "review"   # always auto-review generated code


# ── graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    # nodes
    graph.add_node("router",           RouterNode)
    graph.add_node("context_retriever", ContextRetrieverNode)
    graph.add_node("code_agent",       CodeAgentNode)
    graph.add_node("review_agent",     ReviewAgentNode)
    graph.add_node("simple_agent",     SimpleAgentNode)
    graph.add_node("complex_agent",    ComplexAgentNode)

    # edges
    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieve_then_code":    "context_retriever",
            "retrieve_then_complex": "context_retriever",
            "review":                "review_agent",
            "simple":                "simple_agent",
        },
    )

    # after retrieval, branch to code or complex
    graph.add_conditional_edges(
        "context_retriever",
        lambda s: "code" if s["category"] == "CODE" else "complex",
        {
            "code":    "code_agent",
            "complex": "complex_agent",
        },
    )

    graph.add_edge("code_agent",    "review_agent")
    graph.add_edge("review_agent",  END)
    graph.add_edge("simple_agent",  END)
    graph.add_edge("complex_agent", END)

    return graph.compile()


# ── runner ────────────────────────────────────────────────────────────────────

_graph = None


def run_pipeline(task: str) -> str:
    global _graph
    if _graph is None:
        _graph = build_graph()

    initial_state: PipelineState = {
        "task": task,
        "category": "",
        "context": "",
        "code_output": "",
        "review_output": "",
        "final_answer": "",
    }

    final_state = _graph.invoke(initial_state)

    # render output
    if final_state["category"] == "CODE":
        console.print(
            Panel(
                Markdown(final_state["code_output"]),
                title="✅ Generated code",
                border_style="green",
            )
        )
        console.print(
            Panel(
                Markdown(final_state["review_output"]),
                title="🔍 Auto-review",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                Markdown(final_state["final_answer"]),
                title="✅ Answer",
                border_style="blue",
            )
        )

    return final_state["final_answer"]


# ── interactive CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print(
        Panel(
            "[bold green]🤖 AI Code Pipeline with RAG[/bold green]\n"
            "Type [bold]exit[/bold] to quit\n"
            "Tip: run [bold]python -m indexer.index_docs ./sample_docs[/bold] first",
            border_style="green",
        )
    )
    while True:
        task = console.input("\n[bold cyan]Your task:[/bold cyan] ").strip()
        if task.lower() in ("exit", "quit"):
            break
        if task:
            run_pipeline(task)