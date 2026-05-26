"""Fusion retrieval: combines query-text similarity, taste alignment,
metadata keyword match, and image similarity.

`use_taste` and `taste_key` are the levers the agent and the eval script
pull to switch behaviour between system variants.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from . import config
from .taste import load_centroids

# Cached singletons.
_text_model: SentenceTransformer | None = None
_image_model: SentenceTransformer | None = None
_centroids: dict[str, np.ndarray] | None = None


def _get_text_model() -> SentenceTransformer:
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer(config.TEXT_EMBED_MODEL)
    return _text_model


def _get_image_model() -> SentenceTransformer:
    global _image_model
    if _image_model is None:
        _image_model = SentenceTransformer(config.IMAGE_EMBED_MODEL)
    return _image_model


def _get_centroids() -> dict[str, np.ndarray]:
    global _centroids
    if _centroids is None:
        _centroids = load_centroids()
    return _centroids


def _keyword_match_score(query: str, meta: dict) -> float:
    """Bag-of-words overlap between query tokens and metadata + visual_description fields."""
    import re
    q_tokens = {re.sub(r"[^\w]", "", t.lower()) for t in query.split() if len(t) > 2}
    q_tokens.discard("")
    if not q_tokens:
        return 0.0
    fields = " ".join(
        str(meta.get(k, "")) for k in
        ("name", "type", "era", "faction", "mood", "canon_status", "medium", "visual_description")
    ).lower()
    field_tokens = {re.sub(r"[^\w]", "", t) for t in fields.split()}
    field_tokens.discard("")
    overlap = len(q_tokens & field_tokens)
    return overlap / max(1, len(q_tokens))


def _taste_alignment(doc_emb: np.ndarray, centroid: np.ndarray) -> float:
    return float(np.dot(doc_emb, centroid))


@dataclass
class RetrievalResult:
    entity_id: str
    metadata: dict[str, Any]
    final_score: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalTrace:
    results: list[RetrievalResult]
    use_taste: bool
    taste_key: str | None
    latency_ms: float


def retrieve(
    query: str,
    *,
    use_taste: bool = False,
    taste_key: str = "overall",
    top_k: int = config.TOP_K_FINAL,
    image_query_path: str | None = None,
    intent: str | None = None,
) -> RetrievalTrace:
    """Fused retrieval. taste_key is one of: overall | mood_tragic | mood_epic | mood_political | mood_cathartic.
    intent is used to adjust fusion weights for similarity queries (boosts visual/keyword channels)."""
    t0 = time.perf_counter()

    # Adjust fusion weights for visual/similarity queries — same selectivity principle
    # applied to modality weights as to taste: don't apply text-heavy weights when the
    # user is querying by appearance.
    alpha = config.ALPHA
    gamma = config.GAMMA
    delta = config.DELTA
    if intent == "similarity":
        alpha = 0.35
        gamma = 0.30
        delta = 0.10

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    text_coll = client.get_collection(config.TEXT_COLLECTION)

    # 1. Top-N candidates from text collection.
    text_model = _get_text_model()
    q_emb = text_model.encode([query], normalize_embeddings=True)[0]
    text_res = text_coll.query(
        query_embeddings=[q_emb.tolist()],
        n_results=config.TOP_K_CANDIDATES,
        include=["embeddings", "metadatas", "distances"],
    )

    # 2. Image similarity — also collect CLIP-only candidates not in text top-N.
    image_sims: dict[str, float] = {}
    clip_only_ids: list[str] = []
    try:
        image_coll = client.get_collection(config.IMAGE_COLLECTION)
        img_model = _get_image_model()
        if image_query_path is not None:
            from PIL import Image as PILImage
            img = PILImage.open(image_query_path).convert("RGB")
            img_q = img_model.encode([img], normalize_embeddings=True)[0]
        else:
            img_q = img_model.encode([query], normalize_embeddings=True)[0]
        img_res = image_coll.query(
            query_embeddings=[img_q.tolist()],
            n_results=config.TOP_K_CANDIDATES,
            include=["distances"],
        )
        text_ids = set(text_res["ids"][0])
        for eid, dist in zip(img_res["ids"][0], img_res["distances"][0]):
            image_sims[eid] = 1.0 - dist
            if eid not in text_ids:
                clip_only_ids.append(eid)
    except Exception as e:
        print(f"[retrieve] image query failed: {e}")

    # 2b. Fetch text embeddings + metadata for CLIP-only candidates so they
    #     participate in fusion even when text search missed them.
    extra_embs: dict[str, np.ndarray] = {}
    extra_metas: dict[str, dict] = {}
    extra_sims: dict[str, float] = {}
    if clip_only_ids:
        extra = text_coll.get(ids=clip_only_ids, include=["embeddings", "metadatas"])
        for eid, emb, meta in zip(extra["ids"], extra["embeddings"], extra["metadatas"]):
            arr = np.array(emb)
            extra_embs[eid] = arr
            extra_metas[eid] = meta
            extra_sims[eid] = float(np.dot(q_emb, arr))

    # 3. Taste centroid.
    centroid = None
    if use_taste:
        centroids = _get_centroids()
        centroid = centroids.get(taste_key, centroids.get("overall"))

    # 4. Fusion over text candidates + CLIP-only candidates.
    candidates: list[RetrievalResult] = []

    for eid, doc_emb, meta, dist in zip(
        text_res["ids"][0],
        text_res["embeddings"][0],
        text_res["metadatas"][0],
        text_res["distances"][0],
    ):
        query_sim = 1.0 - dist
        meta_score = _keyword_match_score(query, meta)
        img_sim = image_sims.get(eid, 0.0)
        taste_align = _taste_alignment(np.array(doc_emb), centroid) if centroid is not None else 0.0
        final = (
            alpha * query_sim
            + config.BETA * taste_align * (1.0 if use_taste else 0.0)
            + gamma * meta_score
            + delta * img_sim
        )
        candidates.append(RetrievalResult(
            entity_id=eid, metadata=meta, final_score=final,
            components={"query_sim": query_sim, "taste_align": taste_align,
                        "meta_score": meta_score, "image_sim": img_sim},
        ))

    for eid in clip_only_ids:
        doc_emb = extra_embs[eid]
        meta = extra_metas[eid]
        query_sim = extra_sims[eid]
        img_sim = image_sims.get(eid, 0.0)
        meta_score = _keyword_match_score(query, meta)
        taste_align = _taste_alignment(doc_emb, centroid) if centroid is not None else 0.0
        final = (
            alpha * query_sim
            + config.BETA * taste_align * (1.0 if use_taste else 0.0)
            + gamma * meta_score
            + delta * img_sim
        )
        candidates.append(RetrievalResult(
            entity_id=eid, metadata=meta, final_score=final,
            components={"query_sim": query_sim, "taste_align": taste_align,
                        "meta_score": meta_score, "image_sim": img_sim},
        ))

    candidates.sort(key=lambda r: r.final_score, reverse=True)
    top = candidates[:top_k]
    latency_ms = (time.perf_counter() - t0) * 1000
    return RetrievalTrace(
        results=top,
        use_taste=use_taste,
        taste_key=taste_key if use_taste else None,
        latency_ms=latency_ms,
    )


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "a deeply tragic character arc"
    trace = retrieve(q, use_taste=True, taste_key="mood_tragic")
    print(f"\nQuery: {q}")
    print(f"use_taste={trace.use_taste} taste_key={trace.taste_key} latency={trace.latency_ms:.1f}ms\n")
    for i, r in enumerate(trace.results, 1):
        print(f"{i}. {r.metadata.get('name')} — score={r.final_score:.3f}")
