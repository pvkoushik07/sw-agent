"""LangGraph agent: classify → resolve_references → retrieve → synthesise → END

Run a single query:
    python -m src.agent "a deeply tragic character arc"

Run with conversation history (Python API):
    from src.agent import run_agent
    out1 = run_agent("List Jedi from the prequel era")
    out2 = run_agent("Which of those did I rate highest?", history=out1["history"])
"""
from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from . import config, llm
from .retrieve import RetrievalResult, retrieve

REFERENCE_PHRASES = [
    "those", "of these", "from before", "the previous ones", "that list",
    "earlier", "above", "you mentioned", "among those", "of those",
    "which of those", "which of them", "from that list", "the ones you",
    "the results", "those results",
]


class AgentState(TypedDict, total=False):
    query: str
    original_query: str          # preserved if resolve_references rewrites query
    resolved_query: str | None   # the rewritten query (None if no reference)
    reference_detected: bool
    intent: str
    intent_confidence: float
    use_taste: bool
    taste_key: str | None
    results: list[RetrievalResult]
    answer: str
    trace: dict[str, Any]
    history: list[dict]          # [{query, intent, retrieved_ids, retrieved_names, answer}]
    max_history_turns: int


# ── Nodes ──────────────────────────────────────────────────────────────────────

def classify_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    out = llm.classify_intent(state["query"])
    intent = out["intent"]

    if intent == "mood_tragic":
        use_taste, taste_key = True, "mood_tragic"
    elif intent == "mood_epic":
        use_taste, taste_key = True, "mood_epic"
    elif intent == "mood_political":
        use_taste, taste_key = True, "mood_political"
    elif intent == "mood_cathartic":
        use_taste, taste_key = True, "mood_cathartic"
    elif intent == "mood_goofy":
        use_taste, taste_key = True, "mood_goofy"
    elif intent == "mood_general":
        use_taste, taste_key = True, "overall"
    else:
        use_taste, taste_key = False, None

    state["intent"] = intent
    state["intent_confidence"] = out["confidence"]
    state["use_taste"] = use_taste
    state["taste_key"] = taste_key
    state.setdefault("trace", {})["classify_ms"] = (time.perf_counter() - t0) * 1000
    return state


def resolve_references_node(state: AgentState) -> AgentState:
    """Rewrite referential queries using last turn's retrieved entities."""
    t0 = time.perf_counter()
    state["original_query"] = state["query"]
    state["reference_detected"] = False
    state["resolved_query"] = None

    history = state.get("history", [])
    if not history:
        state.setdefault("trace", {})["resolve_ms"] = (time.perf_counter() - t0) * 1000
        return state

    q_lower = state["query"].lower()
    if not any(phrase in q_lower for phrase in REFERENCE_PHRASES):
        state.setdefault("trace", {})["resolve_ms"] = (time.perf_counter() - t0) * 1000
        return state

    # Reference phrase detected — ask LLM to produce a self-contained query
    out = llm.resolve_query_reference(state["query"], history[-1])
    if out["has_reference"]:
        state["reference_detected"] = True
        state["resolved_query"] = out["rewritten_query"]
        state["query"] = out["rewritten_query"]

    state.setdefault("trace", {})["resolve_ms"] = (time.perf_counter() - t0) * 1000
    return state


def retrieve_node(state: AgentState) -> AgentState:
    trace = retrieve(
        state["query"],
        use_taste=state.get("use_taste", False),
        taste_key=state.get("taste_key") or "overall",
        top_k=config.TOP_K_FINAL,
    )
    state["results"] = trace.results
    state.setdefault("trace", {})["retrieve_ms"] = trace.latency_ms
    return state


def synthesise_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    candidates = [r.metadata for r in state.get("results", [])]
    state["answer"] = llm.synthesise_answer(state["query"], candidates)
    state.setdefault("trace", {})["synthesise_ms"] = (time.perf_counter() - t0) * 1000
    return state


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_agent():
    g = StateGraph(AgentState)
    g.add_node("classify", classify_node)
    g.add_node("resolve_references", resolve_references_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("synthesise", synthesise_node)
    g.set_entry_point("classify")
    g.add_edge("classify", "resolve_references")
    g.add_edge("resolve_references", "retrieve")
    g.add_edge("retrieve", "synthesise")
    g.add_edge("synthesise", END)
    return g.compile()


_compiled = None


def run_agent(query: str, history: list[dict] | None = None) -> AgentState:
    global _compiled
    if _compiled is None:
        _compiled = build_agent()

    init_history = list(history) if history else []
    t0 = time.perf_counter()
    out = _compiled.invoke({
        "query": query,
        "history": init_history,
        "max_history_turns": 5,
    })
    out.setdefault("trace", {})["total_ms"] = (time.perf_counter() - t0) * 1000

    # Append this turn to history (outside the graph to avoid a 5th node)
    results = out.get("results", [])
    new_turn = {
        "query": out.get("original_query", query),
        "intent": out.get("intent"),
        "retrieved_ids": [r.entity_id for r in results],
        "retrieved_names": [r.metadata.get("name", r.entity_id) for r in results],
        "answer": out.get("answer", ""),
    }
    max_turns = out.get("max_history_turns", 5)
    out["history"] = (init_history + [new_turn])[-max_turns:]

    return out


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "a deeply tragic character arc"
    out = run_agent(q)
    print(f"\nQuery: {q}")
    print(f"Intent: {out['intent']} (conf={out.get('intent_confidence', 0):.2f})")
    print(f"use_taste={out.get('use_taste')} taste_key={out.get('taste_key')}")
    if out.get("reference_detected"):
        print(f"Reference rewritten to: {out.get('resolved_query')}")
    print("\nTop results:")
    for i, r in enumerate(out.get("results", []), 1):
        print(f"  {i}. {r.metadata.get('name')} (score={r.final_score:.3f})")
    print(f"\nAnswer:\n  {out.get('answer')}\n")
    print(f"Trace: {out.get('trace')}")
