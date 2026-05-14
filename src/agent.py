"""LangGraph agent: classify_intent -> retrieve -> synthesise -> answer.

Run a single query: python -m src.agent "a deeply tragic character arc"
"""
from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from . import config, llm
from .retrieve import RetrievalResult, retrieve


class AgentState(TypedDict, total=False):
    query: str
    intent: str
    intent_confidence: float
    use_taste: bool
    taste_key: str | None
    results: list[RetrievalResult]
    answer: str
    trace: dict[str, Any]


# ----------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------

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
    else:  # factual, similarity, comparative
        use_taste, taste_key = False, None

    state["intent"] = intent
    state["intent_confidence"] = out["confidence"]
    state["use_taste"] = use_taste
    state["taste_key"] = taste_key
    state.setdefault("trace", {})["classify_ms"] = (time.perf_counter() - t0) * 1000
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


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------

def build_agent():
    g = StateGraph(AgentState)
    g.add_node("classify", classify_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("synthesise", synthesise_node)
    g.set_entry_point("classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "synthesise")
    g.add_edge("synthesise", END)
    return g.compile()


_compiled = None

def run_agent(query: str) -> AgentState:
    global _compiled
    if _compiled is None:
        _compiled = build_agent()
    t0 = time.perf_counter()
    out = _compiled.invoke({"query": query})
    out.setdefault("trace", {})["total_ms"] = (time.perf_counter() - t0) * 1000
    return out


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "a deeply tragic character arc"
    out = run_agent(q)
    print(f"\nQuery: {q}")
    print(f"Intent: {out['intent']} (conf={out.get('intent_confidence', 0):.2f})")
    print(f"use_taste={out.get('use_taste')} taste_key={out.get('taste_key')}\n")
    print("Top results:")
    for i, r in enumerate(out.get("results", []), 1):
        print(f"  {i}. {r.metadata.get('name')} (score={r.final_score:.3f})")
    print(f"\nAnswer:\n  {out.get('answer')}\n")
    print(f"Trace: {out.get('trace')}")
