"""Evaluate 5 systems on the test set. Run: python -m eval.evaluate

Systems:
  S1 plain_llm       — Gemini, no retrieval
  S2 hybrid_no_taste — Hybrid retrieval, no taste, no agent
  S3 single_taste    — Hybrid + single 'overall' taste centroid always on
  S4 mood_no_router  — Mood centroids applied uniformly (no intent gating)
  S5 full_agent      — Router + mood centroids (proposed)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass

import pandas as pd

from src import config, llm
from src.agent import run_agent
from src.retrieve import retrieve


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & gold_ids) / len(gold_ids)


def hit_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    return 1.0 if set(retrieved_ids[:k]) & gold_ids else 0.0


def mrr(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    for i, eid in enumerate(retrieved_ids, 1):
        if eid in gold_ids:
            return 1.0 / i
    return 0.0


@dataclass
class SystemOutput:
    retrieved_ids: list[str]
    answer: str
    latency_ms: float
    intent: str | None = None


def system_plain_llm(query: str) -> SystemOutput:
    t0 = time.perf_counter()
    answer = llm.synthesise_answer(query, [])
    return SystemOutput(retrieved_ids=[], answer=answer,
                        latency_ms=(time.perf_counter() - t0) * 1000)


def system_hybrid_no_taste(query: str) -> SystemOutput:
    trace = retrieve(query, use_taste=False)
    answer = llm.synthesise_answer(query, [r.metadata for r in trace.results])
    return SystemOutput(
        retrieved_ids=[r.entity_id for r in trace.results],
        answer=answer,
        latency_ms=trace.latency_ms,
    )


def system_single_taste(query: str) -> SystemOutput:
    trace = retrieve(query, use_taste=True, taste_key="overall")
    answer = llm.synthesise_answer(query, [r.metadata for r in trace.results])
    return SystemOutput(
        retrieved_ids=[r.entity_id for r in trace.results],
        answer=answer,
        latency_ms=trace.latency_ms,
    )


def system_mood_no_router(query: str) -> SystemOutput:
    """Mood centroid applied uniformly. Default to mood_tragic for everything
    — the 'always-on mood' ablation isolates the router's contribution."""
    trace = retrieve(query, use_taste=True, taste_key="mood_tragic")
    answer = llm.synthesise_answer(query, [r.metadata for r in trace.results])
    return SystemOutput(
        retrieved_ids=[r.entity_id for r in trace.results],
        answer=answer,
        latency_ms=trace.latency_ms,
    )


def system_full_agent(query: str) -> SystemOutput:
    out = run_agent(query)
    return SystemOutput(
        retrieved_ids=[r.entity_id for r in out.get("results", [])],
        answer=out.get("answer", ""),
        latency_ms=out.get("trace", {}).get("total_ms", 0),
        intent=out.get("intent"),
    )


SYSTEMS = {
    "S1_plain_llm": system_plain_llm,
    "S2_hybrid_no_taste": system_hybrid_no_taste,
    "S3_single_taste": system_single_taste,
    "S4_mood_no_router": system_mood_no_router,
    "S5_full_agent": system_full_agent,
}


def main() -> None:
    if not config.TEST_SET_PATH.exists():
        sys.exit(f"[eval] {config.TEST_SET_PATH} not found.")

    with open(config.TEST_SET_PATH) as f:
        test_set = json.load(f)
    print(f"[eval] loaded {len(test_set)} test cases")

    rows = []
    for case in test_set:
        qid = case["id"]
        query = case["query"]
        family = case["family"]
        gold = set(case["relevant_entity_ids"])
        expected_intent = case.get("expected_intent")

        for sys_name, sys_fn in SYSTEMS.items():
            print(f"[eval] {qid} | {family} | {sys_name}")
            try:
                out = sys_fn(query)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            ground = {"score": 0, "reasoning": "n/a"}
            if sys_name != "S1_plain_llm" and out.retrieved_ids:
                ctx = "\n".join(out.retrieved_ids[:5])
                ground = llm.judge_groundedness(query, ctx, out.answer)

            rows.append({
                "qid": qid,
                "family": family,
                "system": sys_name,
                "query": query,
                "gold_ids": list(gold),
                "retrieved_ids": out.retrieved_ids,
                "answer": out.answer,
                "latency_ms": out.latency_ms,
                "recall_at_5": recall_at_k(out.retrieved_ids, gold, 5),
                "hit_at_3": hit_at_k(out.retrieved_ids, gold, 3),
                "mrr": mrr(out.retrieved_ids, gold),
                "groundedness": ground["score"],
                "intent_predicted": out.intent,
                "intent_expected": expected_intent,
                "intent_correct": (
                    int(out.intent == expected_intent)
                    if out.intent and expected_intent else None
                ),
            })

    df = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_DIR / "raw_results.csv", index=False)
    print(f"[eval] raw results -> {config.RESULTS_DIR / 'raw_results.csv'}")

    summary = df.groupby(["system", "family"]).agg(
        recall_at_5=("recall_at_5", "mean"),
        hit_at_3=("hit_at_3", "mean"),
        mrr=("mrr", "mean"),
        latency_ms=("latency_ms", "mean"),
        groundedness=("groundedness", "mean"),
    ).reset_index()
    summary.to_csv(config.RESULTS_DIR / "summary_by_family.csv", index=False)
    print(f"[eval] family summary -> {config.RESULTS_DIR / 'summary_by_family.csv'}")

    overall = df.groupby("system").agg(
        recall_at_5=("recall_at_5", "mean"),
        hit_at_3=("hit_at_3", "mean"),
        mrr=("mrr", "mean"),
        latency_ms=("latency_ms", "mean"),
        groundedness=("groundedness", "mean"),
    ).reset_index()
    overall.to_csv(config.RESULTS_DIR / "summary_overall.csv", index=False)
    print(f"[eval] overall summary -> {config.RESULTS_DIR / 'summary_overall.csv'}")

    s5 = df[df["system"] == "S5_full_agent"].dropna(subset=["intent_correct"])
    if not s5.empty:
        acc = s5["intent_correct"].mean()
        print(f"[eval] S5 router accuracy: {acc:.1%} ({int(s5['intent_correct'].sum())}/{len(s5)})")

    # ------------------------------------------------------------------
    # Drift analysis — tests claim (ii) of the hypothesis:
    # always-on taste should DISPLACE correct factual answers.
    # We measure per-query the entities that appear in S2 (no taste) top-5
    # but get pushed OUT of top-5 in S3/S4 (always-on taste). If those
    # displaced entities are gold, that's measurable harm caused by taste.
    # ------------------------------------------------------------------
    print("\n[eval] computing drift analysis (claim ii)")
    drift_rows = []
    for qid in df["qid"].unique():
        rows_for_q = df[df["qid"] == qid]
        s2 = rows_for_q[rows_for_q["system"] == "S2_hybrid_no_taste"]
        if s2.empty:
            continue
        s2_top5 = set(s2.iloc[0]["retrieved_ids"][:5])
        gold = set(s2.iloc[0]["gold_ids"])
        family = s2.iloc[0]["family"]

        for sys_name in ("S3_single_taste", "S4_mood_no_router", "S5_full_agent"):
            other = rows_for_q[rows_for_q["system"] == sys_name]
            if other.empty:
                continue
            other_top5 = set(other.iloc[0]["retrieved_ids"][:5])
            displaced = s2_top5 - other_top5  # in S2 top5 but not in other top5
            gold_displaced = displaced & gold  # of those, how many were correct?
            drift_rows.append({
                "qid": qid,
                "family": family,
                "system_vs_S2": sys_name,
                "n_displaced": len(displaced),
                "n_gold_displaced": len(gold_displaced),
                "gold_displaced_ids": list(gold_displaced),
            })

    drift_df = pd.DataFrame(drift_rows)
    drift_df.to_csv(config.RESULTS_DIR / "drift_analysis.csv", index=False)
    print(f"[eval] drift analysis -> {config.RESULTS_DIR / 'drift_analysis.csv'}")

    # Aggregate drift by family — this is the key table for claim (ii).
    drift_summary = drift_df.groupby(["system_vs_S2", "family"]).agg(
        avg_displaced=("n_displaced", "mean"),
        avg_gold_displaced=("n_gold_displaced", "mean"),
        total_gold_displaced=("n_gold_displaced", "sum"),
    ).reset_index()
    drift_summary.to_csv(config.RESULTS_DIR / "drift_summary.csv", index=False)
    print(f"[eval] drift summary -> {config.RESULTS_DIR / 'drift_summary.csv'}")
    print("\nDrift summary (avg gold entities displaced from top-5 vs S2 baseline):")
    print(drift_summary.to_string(index=False))
    print("\n** For claim (ii), we expect S3 and S4 to show positive avg_gold_displaced")
    print("   on FACTUAL queries — meaning always-on taste pushed correct answers out.")
    print("   S5 should show near-zero drift on factual queries (router protects). **")

    print("\nOverall results:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
