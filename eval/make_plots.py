"""Generate report figures 3, 4, 5 from eval result CSVs.

Run AFTER `python -m eval.evaluate` has produced the summary CSVs.
Usage: python -m eval.make_plots
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src import config


SYSTEM_ORDER = [
    "S1_plain_llm",
    "S2_hybrid_no_taste",
    "S3_single_taste",
    "S4_mood_no_router",
    "S5_full_agent",
]
SYSTEM_LABELS = {
    "S1_plain_llm": "S1: plain LLM",
    "S2_hybrid_no_taste": "S2: hybrid, no taste",
    "S3_single_taste": "S3: single taste",
    "S4_mood_no_router": "S4: mood, no router",
    "S5_full_agent": "S5: full agent",
}
FAMILY_ORDER = ["factual", "cross_modal", "multi_hop", "ambiguous_personalised"]


def figure3_recall_by_family() -> None:
    """Grouped bar chart: Recall@5 across systems × families."""
    df = pd.read_csv(config.RESULTS_DIR / "summary_by_family.csv")
    pivot = df.pivot(index="family", columns="system", values="recall_at_5")
    pivot = pivot.reindex(FAMILY_ORDER)[SYSTEM_ORDER]

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=ax, width=0.8,
               color=["#aaaaaa", "#4477aa", "#ee9933", "#bb5566", "#117733"])
    ax.set_ylabel("Recall@5")
    ax.set_title("Recall@5 by system and query family")
    ax.set_xlabel("")
    ax.set_xticklabels(FAMILY_ORDER, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend([SYSTEM_LABELS[s] for s in SYSTEM_ORDER],
              loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False)
    plt.tight_layout()
    out = config.RESULTS_DIR / "fig3_recall_by_family.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plots] -> {out}")


def figure4_drift_factual() -> None:
    """Bar chart: avg gold entities displaced from top-5 on factual queries.

    Visualises claim (ii): always-on taste pulls correct factual answers out
    of the top-5. S5's router should restore the result to ~zero displacement.
    """
    df = pd.read_csv(config.RESULTS_DIR / "drift_summary.csv")
    factual = df[df["family"] == "factual"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [SYSTEM_LABELS[s] for s in factual["system_vs_S2"]],
        factual["avg_gold_displaced"],
        color=["#ee9933", "#bb5566", "#117733"],
    )
    ax.set_ylabel("Avg gold entities displaced from top-5 (vs S2 baseline)")
    ax.set_title("Drift on factual queries — claim (ii) test")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, factual["avg_gold_displaced"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", fontsize=10)
    plt.tight_layout()
    out = config.RESULTS_DIR / "fig4_drift_factual.png"
    plt.savefig(out, dpi=150)
    print(f"[plots] -> {out}")


def figure5_latency() -> None:
    """Bar chart: mean latency per system. Annotate with peer-project ~12s."""
    df = pd.read_csv(config.RESULTS_DIR / "summary_overall.csv")
    df = df.set_index("system").reindex(SYSTEM_ORDER).reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        [SYSTEM_LABELS[s] for s in df["system"]],
        df["latency_ms"] / 1000.0,
        color=["#aaaaaa", "#4477aa", "#ee9933", "#bb5566", "#117733"],
    )
    ax.set_ylabel("Mean latency per query (s)")
    ax.set_title("Latency comparison — peer project with always-on LLM rerank: ~12s")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, df["latency_ms"] / 1000.0):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                f"{val:.2f}s", ha="center", fontsize=10)
    ax.set_xticklabels([SYSTEM_LABELS[s] for s in df["system"]], rotation=15, ha="right")
    plt.tight_layout()
    out = config.RESULTS_DIR / "fig5_latency.png"
    plt.savefig(out, dpi=150)
    print(f"[plots] -> {out}")


def figure6_modality_ablation() -> None:
    """Grouped bar chart: Recall@5 by channel ablation × query family."""
    path = config.RESULTS_DIR / "modality_summary_by_family.csv"
    if not path.exists():
        print(f"[plots] skipping fig6 — {path} not found (run eval.evaluate_modality first)")
        return

    df = pd.read_csv(path)
    MODALITY_ORDER = ["S6_text_only", "S7_image_only", "S8_full_sanity"]
    MODALITY_LABELS = {
        "S6_text_only":   "S6: text only",
        "S7_image_only":  "S7: image only",
        "S8_full_sanity": "S8: full (sanity)",
    }
    MODALITY_COLORS = ["#4477aa", "#ee7733", "#228833"]

    pivot = df.pivot(index="family", columns="system", values="recall_at_5")
    # Only keep the three modality systems; fill missing with 0
    for s in MODALITY_ORDER:
        if s not in pivot.columns:
            pivot[s] = 0.0
    pivot = pivot.reindex(FAMILY_ORDER)[MODALITY_ORDER]

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=ax, width=0.7, color=MODALITY_COLORS)
    ax.set_ylabel("Recall@5")
    ax.set_title("Modality ablation: Recall@5 by channel and query family")
    ax.set_xlabel("")
    ax.set_xticklabels(FAMILY_ORDER, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        [MODALITY_LABELS[s] for s in MODALITY_ORDER],
        loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False,
    )
    plt.tight_layout()
    out = config.RESULTS_DIR / "fig6_modality_ablation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plots] -> {out}")


def figure7_hit3_by_family() -> None:
    """Grouped bar chart: Hit@3 across systems × families."""
    df = pd.read_csv(config.RESULTS_DIR / "summary_by_family.csv")
    pivot = df.pivot(index="family", columns="system", values="hit_at_3")
    pivot = pivot.reindex(FAMILY_ORDER)[SYSTEM_ORDER]

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=ax, width=0.8,
               color=["#aaaaaa", "#4477aa", "#ee9933", "#bb5566", "#117733"])
    ax.set_ylabel("Hit@3")
    ax.set_title("Hit@3 by system and query family")
    ax.set_xlabel("")
    ax.set_xticklabels(FAMILY_ORDER, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend([SYSTEM_LABELS[s] for s in SYSTEM_ORDER],
              loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False)
    plt.tight_layout()
    out = config.RESULTS_DIR / "fig7_hit3_by_family.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plots] -> {out}")


def main() -> None:
    figure3_recall_by_family()
    figure4_drift_factual()
    figure5_latency()
    figure6_modality_ablation()
    figure7_hit3_by_family()
    print("[plots] all figures generated in eval/results/")


if __name__ == "__main__":
    main()
