"""Ingest entities.csv into two ChromaDB collections (text + image).

Run: python -m src.ingest
"""
from __future__ import annotations

import sys
from pathlib import Path

import chromadb
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer

from . import config


def _load_entities() -> pd.DataFrame:
    if not config.ENTITIES_CSV.exists():
        sys.exit(f"[ingest] {config.ENTITIES_CSV} not found.")
    df = pd.read_csv(config.ENTITIES_CSV)
    required = {
        "entity_id", "name", "type", "era", "faction", "first_appearance",
        "medium", "canon_status", "description", "your_take",
        "your_rating", "mood", "visual_description", "image_note",
    }
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ingest] entities.csv missing columns: {missing}")
    if df["entity_id"].duplicated().any():
        sys.exit("[ingest] duplicate entity_id found")
    print(f"[ingest] loaded {len(df)} entities")
    return df


def _build_text_doc(row: pd.Series) -> str:
    """Combined text used for the main retrieval index."""
    return (
        f"{row['name']}. Type: {row['type']}. Era: {row['era']}. "
        f"Faction: {row['faction']}. Mood: {row['mood']}. "
        f"Description: {row['description']} "
        f"Visual: {row['visual_description']} "
        f"User's take: {row['your_take']}"
    )


def _check_image(entity_id: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = config.IMAGES_DIR / f"{entity_id}{ext}"
        if p.exists():
            return p
    return None


def ingest() -> None:
    df = _load_entities()
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    # Drop existing collections for a clean rebuild.
    for name in (config.TEXT_COLLECTION, config.IMAGE_COLLECTION):
        try:
            client.delete_collection(name)
            print(f"[ingest] dropped existing collection {name}")
        except Exception:
            pass

    # --- TEXT collection ---
    print(f"[ingest] embedding text with {config.TEXT_EMBED_MODEL}")
    text_model = SentenceTransformer(config.TEXT_EMBED_MODEL)
    text_coll = client.create_collection(
        name=config.TEXT_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [_build_text_doc(row) for _, row in df.iterrows()]
    text_embs = text_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    metadatas = df.to_dict(orient="records")
    for m in metadatas:
        m["your_rating"] = int(m["your_rating"])

    text_coll.add(
        ids=df["entity_id"].tolist(),
        embeddings=text_embs.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    print(f"[ingest] text collection: {text_coll.count()} entries")

    # --- IMAGE collection ---
    print(f"[ingest] embedding images with {config.IMAGE_EMBED_MODEL}")
    image_model = SentenceTransformer(config.IMAGE_EMBED_MODEL)
    image_coll = client.create_collection(
        name=config.IMAGE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    ids, images, image_metas, missing = [], [], [], []
    for (_, row), meta in zip(df.iterrows(), metadatas):
        img_path = _check_image(row["entity_id"])
        if img_path is None:
            missing.append(row["entity_id"])
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[ingest] failed to open {img_path}: {e}")
            continue
        ids.append(row["entity_id"])
        images.append(img)
        image_metas.append(meta)

    if missing:
        print(f"[ingest] WARNING: {len(missing)} entities have no image:")
        for m in missing:
            print(f"  - {m}")
        print("  Add images to data/images/{entity_id}.jpg to include them in CLIP retrieval.")

    if images:
        img_embs = image_model.encode(images, show_progress_bar=True, normalize_embeddings=True)
        image_coll.add(
            ids=ids,
            embeddings=img_embs.tolist(),
            metadatas=image_metas,
        )
    print(f"[ingest] image collection: {image_coll.count()} entries")
    print("[ingest] done.")


if __name__ == "__main__":
    ingest()
