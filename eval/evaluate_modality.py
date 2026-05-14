"""Modality ablation: which retrieval channel contributes what.

Three systems, all using the S5 router + mood centroids, but with
fusion weights patched to isolate individual channels:

  S6  text_only  — DELTA=0   (image channel silenced)
  S7  image_only — ALPHA=0, GAMMA=0   (text + meta channels silenced)
  S8  full_sanity — default weights (sanity check; should match S5)

Weights are restored via try/finally after every run.
Results written to eval/results/modality_summary_{overall,by_family}.csv.

Run: python -m eval.evaluate_modality
"""
from __future__ import annotations

import contextlib
import json
import sys
from dataclasses import dataclass

import pandas as pd

from src import config
from src.agent import run_agent
from eval.evaluate import recall_at_k, hit_at_k, mrr


# ── Config patcher ─────────────────────────────────────────────────────────────

@contextlib.contextmanager
def patch_config(**kwargs):
    """Temporarily set config attributes, restoring originals in finally."""
    originals = {k: getattr(config, k) for k in kwargs}
    try:
        for k, v in kwargs.items():
            setattr(config, k, v)
        yield
    finally:
        for k, v in originals.items():
            setattr(config, k, v)


def _weights_str() -> str:
    return (
        f"ALPHA={config.ALPHA} BETA={config.BETA} "
        f"GAMMA={config.GAMMA} DELTA={config.DELTA}"
    )


# ── System functions ───────────────────────────────────────────────────────────

def system_s6_text_only(query: str) -> tuple[list[str], str | None]:
    """Text + taste + meta only (DELTA=0 kills image channel)."""
    print(f"    weights before: {_weights_str()}")
    with patch_config(DELTA=0.0):
        print(f"    weights during: {_weights_str()}")
        out = run_agent(query)
    print(f"    weights after:  {_weights_str()}")
    return [r.entity_id for r in out.get("results", [])], out.get("intent")


def system_s7_image_only(query: str) -> tuple[list[str], str | None]:
    """Image + taste only (ALPHA=0, GAMMA=0 kills text and meta channels)."""
    print(f"    weights before: {_weights_str()}")
    with patch_config(ALPHA=0.0, GAMMA=0.0):
        print(f"    weights during: {_weights_str()}")
        out = run_agent(query)
    print(f"    weights after:  {_weights_str()}")
    return [r.entity_id for r in out.get("results", [])], out.get("intent")


def system_s8_full_sanity(query: str) -> tuple[list[str], str | None]:
    """Default weights — should reproduce S5 within Gemini stochasticity."""
    print(f"    weights: {_weights_str()}")
    out = run_agent(query)
    return [r.entity_id for r in out.get("results", [])], out.get("intent")


SYSTEMS = {
    "S6_text_only":    system_s6_text_only,
    "S7_image_only":   system_s7_image_only,
    "S8_full_sanity":  system_s8_full_sanity,
}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not config.TEST_SET_PATH.exists():
        sys.exit(f"[modality] {config.TEST_SET_PATH} not found.")

    with open(config.TEST_SET_PATH) as f:
        test_set = json.load(f)
    # Only evaluate on the original 20 locked queries.
    original = [c for c in test_set if c["family"] != "conversational_followup"]
    print(f"[modality] loaded {len(original)} test cases (original families only)")

    rows = []
    for case in original:
        qid    = case["id"]
        query  = case["query"]
        family = case["family"]
        gold   = set(case["relevant_entity_ids"])

        for sys_name, sys_fn in SYSTEMS.items():
            print(f"[modality] {qid} | {family} | {sys_name}")
            try:
                retrieved, intent = sys_fn(query)
            except Exception as e:
                print(f"  ERROR: {e}")
                # Verify weights restored even on error
                print(f"  weights after error: {_weights_str()}")
                continue

            rows.append({
                "qid":         qid,
                "family":      family,
                "system":      sys_name,
                "query":       query,
                "gold_ids":    list(gold),
                "retrieved_ids": retrieved,
                "intent":      intent,
                "recall_at_5": recall_at_k(retrieved, gold, 5),
                "hit_at_3":    hit_at_k(retrieved, gold, 3),
                "mrr":         mrr(retrieved, gold),
            })

    df = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_DIR / "modality_raw.csv", index=False)

    # Summary by family
    by_family = df.groupby(["system", "family"]).agg(
        recall_at_5=("recall_at_5", "mean"),
        hit_at_3=("hit_at_3", "mean"),
        mrr=("mrr", "mean"),
    ).reset_index()
    by_family.to_csv(config.RESULTS_DIR / "modality_summary_by_family.csv", index=False)
    print(f"\n[modality] family summary -> {config.RESULTS_DIR / 'modality_summary_by_family.csv'}")

    # Summary overall
    overall = df.groupby("system").agg(
        recall_at_5=("recall_at_5", "mean"),
        hit_at_3=("hit_at_3", "mean"),
        mrr=("mrr", "mean"),
    ).reset_index()
    overall.to_csv(config.RESULTS_DIR / "modality_summary_overall.csv", index=False)
    print(f"[modality] overall summary -> {config.RESULTS_DIR / 'modality_summary_overall.csv'}")

    # ── Analysis printout ──────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("(a) modality_summary_by_family.csv")
    print("=" * 60)
    print(by_family.to_string(index=False))

    print("\n" + "=" * 60)
    print("(b) Cross-modal Recall@5 per variant")
    print("=" * 60)
    cross = by_family[by_family["family"] == "cross_modal"][
        ["system", "recall_at_5"]
    ]
    print(cross.to_string(index=False))

    print("\n" + "=" * 60)
    print("(c) S8 sanity check vs main-eval S5 (expected overall Recall@5 ~0.7475)")
    print("=" * 60)
    s5_main = 0.7475
    s8_row = overall[overall["system"] == "S8_full_sanity"]
    if not s8_row.empty:
        s8_recall = s8_row.iloc[0]["recall_at_5"]
        diff = abs(s8_recall - s5_main)
        status = "PASS" if diff <= 0.05 else "FLAG — diff > 0.05, check for weight leakage"
        print(f"  S8 overall Recall@5 : {s8_recall:.4f}")
        print(f"  S5 main-eval        : {s5_main:.4f}")
        print(f"  Difference          : {diff:.4f}  [{status}]")
    else:
        print("  S8 results not found.")

    print("\n" + "=" * 60)
    print("(d) Interpretation")
    print("=" * 60)
    # Find dominant family per system
    for sys_name in ["S6_text_only", "S7_image_only"]:
        rows_sys = by_family[by_family["system"] == sys_name]
        if rows_sys.empty:
            continue
        best_fam = rows_sys.loc[rows_sys["recall_at_5"].idxmax(), "family"]
        worst_fam = rows_sys.loc[rows_sys["recall_at_5"].idxmin(), "family"]
        print(f"  {sys_name}: strongest on '{best_fam}', weakest on '{worst_fam}'")

    # Compare S6 vs S7 on cross_modal specifically
    s6_cross = by_family[
        (by_family["system"] == "S6_text_only") & (by_family["family"] == "cross_modal")
    ]["recall_at_5"].values
    s7_cross = by_family[
        (by_family["system"] == "S7_image_only") & (by_family["family"] == "cross_modal")
    ]["recall_at_5"].values
    if len(s6_cross) and len(s7_cross):
        winner = "text (S6)" if s6_cross[0] >= s7_cross[0] else "image (S7)"
        print(
            f"\n  Cross-modal: text-only={s6_cross[0]:.2f}, image-only={s7_cross[0]:.2f}. "
            f"Dominant channel: {winner}."
        )
        if s6_cross[0] > s7_cross[0]:
            print(
                "  Visual descriptions in the text field carry more signal than raw "
                "CLIP embeddings for this catalogue size."
            )
        else:
            print(
                "  CLIP image similarity provides additional signal beyond what the "
                "visual_description text field alone captures."
            )


if __name__ == "__main__":
    main()
