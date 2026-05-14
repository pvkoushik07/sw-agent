"""Memory ablation: evaluates the conversational_followup family.

Compares turn-2 performance WITH vs WITHOUT the resolve_references node,
by running each FU test case in both modes.

WITH memory:  turn 1 runs first to build history; turn 2 gets that history,
              resolve_references may rewrite the query.

WITHOUT memory: turn 2 runs as a standalone query with empty history;
                resolve_references has nothing to resolve against.

Results written to:
  eval/results/memory_ablation.csv
  eval/results/conversational_followup_summary.csv

Run: python -m eval.evaluate_memory
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from src import config
from src.agent import run_agent
from eval.evaluate import recall_at_k, hit_at_k, mrr


def main() -> None:
    with open(config.TEST_SET_PATH) as f:
        test_set = json.load(f)

    fu_cases = [c for c in test_set if c["family"] == "conversational_followup"]
    if not fu_cases:
        sys.exit("[memory] No conversational_followup cases found in test_set.json.")
    print(f"[memory] {len(fu_cases)} conversational_followup cases")

    rows = []
    traces = []  # detailed trace for the report

    for case in fu_cases:
        qid       = case["id"]
        turn1_q   = case["turn1_query"]
        turn2_q   = case["query"]
        gold      = set(case["relevant_entity_ids"])

        print(f"\n[memory] === {qid} ===")
        print(f"  turn1: {turn1_q}")
        print(f"  turn2: {turn2_q}")
        print(f"  gold:  {gold}")

        # ── WITH memory ────────────────────────────────────────────────────────
        print(f"\n  [with-memory] running turn 1...")
        out1 = run_agent(turn1_q, history=[])
        history_after_t1 = out1["history"]
        t1_names = out1["history"][-1]["retrieved_names"] if history_after_t1 else []
        print(f"  [with-memory] turn 1 retrieved: {t1_names}")

        print(f"  [with-memory] running turn 2 with history...")
        out2_mem = run_agent(turn2_q, history=history_after_t1)
        ref_detected = out2_mem.get("reference_detected", False)
        resolved_q   = out2_mem.get("resolved_query")
        retrieved_mem = [r.entity_id for r in out2_mem.get("results", [])]

        print(f"  [with-memory] reference_detected: {ref_detected}")
        if resolved_q:
            print(f"  [with-memory] rewritten query: {resolved_q}")
        print(f"  [with-memory] retrieved: {retrieved_mem}")

        rows.append({
            "qid": qid,
            "mode": "with_memory",
            "turn1_query": turn1_q,
            "turn2_query": turn2_q,
            "resolved_query": resolved_q or turn2_q,
            "reference_detected": ref_detected,
            "gold_ids": list(gold),
            "retrieved_ids": retrieved_mem,
            "recall_at_5": recall_at_k(retrieved_mem, gold, 5),
            "hit_at_3":    hit_at_k(retrieved_mem, gold, 3),
            "mrr":         mrr(retrieved_mem, gold),
        })

        traces.append({
            "qid": qid,
            "turn1_query": turn1_q,
            "turn1_retrieved_names": t1_names,
            "turn2_original": turn2_q,
            "turn2_resolved": resolved_q,
            "reference_detected": ref_detected,
            "retrieved_with_memory": retrieved_mem,
        })

        # ── WITHOUT memory ─────────────────────────────────────────────────────
        print(f"\n  [no-memory] running turn 2 without history...")
        out2_nomem = run_agent(turn2_q, history=[])
        retrieved_nomem = [r.entity_id for r in out2_nomem.get("results", [])]
        print(f"  [no-memory] retrieved: {retrieved_nomem}")

        rows.append({
            "qid": qid,
            "mode": "no_memory",
            "turn1_query": turn1_q,
            "turn2_query": turn2_q,
            "resolved_query": turn2_q,
            "reference_detected": False,
            "gold_ids": list(gold),
            "retrieved_ids": retrieved_nomem,
            "recall_at_5": recall_at_k(retrieved_nomem, gold, 5),
            "hit_at_3":    hit_at_k(retrieved_nomem, gold, 3),
            "mrr":         mrr(retrieved_nomem, gold),
        })

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "memory_ablation.csv", index=False)
    print(f"\n[memory] ablation -> {config.RESULTS_DIR / 'memory_ablation.csv'}")

    summary = df.groupby("mode").agg(
        recall_at_5=("recall_at_5", "mean"),
        hit_at_3=("hit_at_3", "mean"),
        mrr=("mrr", "mean"),
    ).reset_index()
    summary.to_csv(config.RESULTS_DIR / "conversational_followup_summary.csv", index=False)
    print(f"[memory] summary -> {config.RESULTS_DIR / 'conversational_followup_summary.csv'}")

    # ── Analysis ───────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("(a) Turn-2 Recall@5 and Hit@3: with-memory vs no-memory")
    print("=" * 60)
    per_case = df[["qid", "mode", "recall_at_5", "hit_at_3", "mrr"]]
    print(per_case.to_string(index=False))
    print("\nAggregate:")
    print(summary.to_string(index=False))

    mem_r5   = summary[summary["mode"] == "with_memory"]["recall_at_5"].values[0]
    nomem_r5 = summary[summary["mode"] == "no_memory"]["recall_at_5"].values[0]
    gain_pp  = (mem_r5 - nomem_r5) * 100

    print("\n" + "=" * 60)
    print("(b) Example trace — FU1")
    print("=" * 60)
    fu1 = next((t for t in traces if t["qid"] == "FU1"), None)
    if fu1:
        print(f"  Turn 1 query:          {fu1['turn1_query']}")
        print(f"  Turn 1 retrieved:      {fu1['turn1_retrieved_names']}")
        print(f"  Turn 2 original query: {fu1['turn2_original']}")
        print(f"  Reference detected:    {fu1['reference_detected']}")
        print(f"  Turn 2 rewritten:      {fu1['turn2_resolved'] or '(no rewrite)'}")
        print(f"  Turn 2 retrieved:      {fu1['retrieved_with_memory']}")

    print("\n" + "=" * 60)
    print("(c) Is the memory gain meaningful (>10pp Recall@5)?")
    print("=" * 60)
    print(f"  with_memory Recall@5 : {mem_r5:.4f}")
    print(f"  no_memory   Recall@5 : {nomem_r5:.4f}")
    print(f"  Gain                 : {gain_pp:+.1f}pp")
    if gain_pp > 10:
        print("  MEANINGFUL — memory adds >10pp Recall@5 on conversational queries.")
    elif gain_pp > 0:
        print("  POSITIVE but modest — memory helps but gain is <10pp.")
    else:
        print("  NO GAIN — investigate whether resolve_references fired correctly.")


if __name__ == "__main__":
    main()
