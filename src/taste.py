"""Compute taste centroids from `your_take` embeddings only.

Run: python -m src.taste

Produces:
  - .chroma/taste_centroids.npz  (loaded by retrieve.py)
  - eval/results/taste_pca.png   (visualisation for the report)

IMPORTANT: We embed `your_take` separately here, NOT the combined text used
for retrieval. This keeps the personal-opinion signal pure — the taste vector
is built from the user's voice, not from canon facts mixed in.

Weighting:
    w_i = (rating_i - 5) / 5
So rating=10 -> +1.0, rating=5 -> 0, rating=1 -> -0.8. Negative weights mean
the overall centroid points AWAY from disliked entities.

Mood sub-centroids: uniform mean of entries tagged with that mood AND
rated >= MIN_RATING_FOR_MOOD.
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from . import config


def _load_takes() -> tuple[np.ndarray, pd.DataFrame]:
    """Load entities.csv and embed `your_take` field only."""
    df = pd.read_csv(config.ENTITIES_CSV)
    model = SentenceTransformer(config.TEXT_EMBED_MODEL)
    embs = model.encode(
        df["your_take"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embs, df


def _overall_centroid(embs: np.ndarray, ratings: np.ndarray) -> np.ndarray:
    """Weighted mean using (rating - NEUTRAL)/SCALE as weights."""
    weights = (ratings - config.RATING_NEUTRAL) / config.RATING_SCALE
    if np.abs(weights).sum() < 1e-6:
        return embs.mean(axis=0)
    centroid = (embs * weights[:, None]).sum(axis=0) / np.abs(weights).sum()
    return centroid / np.linalg.norm(centroid)


def _mood_centroid(
    embs: np.ndarray, meta_df: pd.DataFrame, mood: str
) -> np.ndarray | None:
    """Uniform mean over high-rated entries tagged with this mood."""
    mask = (
        meta_df["mood"].str.contains(mood, case=False, na=False)
        & (meta_df["your_rating"] >= config.MIN_RATING_FOR_MOOD)
    )
    n = int(mask.sum())
    if n < config.MIN_GAMES_PER_MOOD:
        print(
            f"[taste] WARNING: mood '{mood}' has only {n} entries rated "
            f">={config.MIN_RATING_FOR_MOOD}. Need >={config.MIN_GAMES_PER_MOOD}."
        )
        if n == 0:
            return None
    centroid = embs[mask.values].mean(axis=0)
    return centroid / np.linalg.norm(centroid)


def compute_centroids() -> dict[str, np.ndarray]:
    embs, meta_df = _load_takes()
    print(f"[taste] loaded {len(embs)} take-embeddings")

    centroids: dict[str, np.ndarray] = {}
    centroids["overall"] = _overall_centroid(embs, meta_df["your_rating"].values)

    for mood in config.MOOD_CENTROIDS:
        c = _mood_centroid(embs, meta_df, mood)
        if c is not None:
            centroids[f"mood_{mood}"] = c

    config.CHROMA_DIR.mkdir(exist_ok=True)
    np.savez(config.CHROMA_DIR / "taste_centroids.npz", **centroids)
    print(f"[taste] saved {len(centroids)} centroids: {list(centroids.keys())}")

    _plot_pca(embs, meta_df, centroids)
    return centroids


def _plot_pca(
    embs: np.ndarray, meta_df: pd.DataFrame, centroids: dict[str, np.ndarray]
) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    centroid_keys = list(centroids.keys())
    centroid_matrix = np.stack([centroids[k] for k in centroid_keys])
    combined = np.vstack([embs, centroid_matrix])

    pca = PCA(n_components=2)
    proj = pca.fit_transform(combined)
    entity_proj = proj[: len(embs)]
    centroid_proj = proj[len(embs):]

    fig, ax = plt.subplots(figsize=(11, 8))
    sc = ax.scatter(
        entity_proj[:, 0],
        entity_proj[:, 1],
        c=meta_df["your_rating"],
        cmap="RdYlGn",
        s=80,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.5,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("User's rating (1-10)")

    markers = {
        "overall": "*",
        "mood_tragic": "v",
        "mood_epic": "^",
        "mood_political": "s",
        "mood_cathartic": "D",
        "mood_goofy": "P",
    }
    for key, xy in zip(centroid_keys, centroid_proj):
        m = markers.get(key, "P")
        ax.scatter(*xy, marker=m, s=400, c="black", edgecolor="white",
                   linewidth=2, label=key, zorder=5)

    ax.set_title("Taste centroids in PCA space (your_take embeddings)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    out = config.RESULTS_DIR / "taste_pca.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"[taste] PCA plot saved to {out}")


def load_centroids() -> dict[str, np.ndarray]:
    """Used by retrieve.py at runtime."""
    path = config.CHROMA_DIR / "taste_centroids.npz"
    if not path.exists():
        sys.exit("[taste] centroids not found. Run `python -m src.taste` first.")
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


if __name__ == "__main__":
    compute_centroids()
