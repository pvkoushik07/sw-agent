# CLAUDE.md — Project Context

> Read this **first** before touching code. This file gives any AI assistant (or future-you after a week away) the context needed to work on this project effectively.

---

## What this project is

**Course:** UQ INFS4205/7205 — Assignment 3 (Personalised Multimodal Agent System, 20 marks)

**Project name:** Taste-Aware Star Wars Universe Agent

**One-line description:** A LangGraph agent that retrieves Star Wars entities (characters, ships, planets, episodes/arcs) from a personally curated catalogue using natural-language queries, fusing a *personal taste vector* — built from the user's own takes and ratings — into retrieval scores to handle subjective queries that generic semantic search fails on.

**Why this is personalised:** The user is a longtime Star Wars fan with specific, sometimes contrarian opinions across films, shows, and Legends. The catalogue's `your_take` field captures these takes in the user's voice (e.g. "the Rogue One hallway scene is the single best minute of Star Wars on film"). The taste vector is derived directly from those takes — not from generic canon summaries.

---

## The hypothesis (this is the whole point — don't drift)

> **Selective personalisation hypothesis.** We hypothesise that personalisation signals in retrieval systems are most valuable when applied *selectively* rather than uniformly. Specifically:
>
> **(i)** a personal taste vector improves *subjective* query performance;
>
> **(ii)** the same vector *degrades* *factual* query performance by pulling results toward the user's favourites — i.e. **personalisation can actively hurt you**;
>
> **(iii)** an intent-classifying router that gates personalisation captures the upside of (i) while eliminating the downside of (ii).
>
> We test all three claims using five system variants on a personally curated 57-entity Star Wars knowledge base, with mood-specific sub-centroids (tragic / epic / political / cathartic) plus an overall centroid.

**Why this framing matters.** The naive framing — "personalisation helps personalisation" — is engineering, not research. It confirms the wiring works but predicts nothing surprising. The selective-personalisation framing puts a non-obvious claim front and centre (ii): adding personalisation can make a retrieval system worse. That's the finding the report sells.

Every design decision must be checked against this hypothesis. If a feature doesn't help us test claims (i), (ii), or (iii), cut it.

**What "success" looks like in the report:**
- **Claim (i):** S5 (full agent) beats S2 (no taste) on `ambiguous_personalised` queries. Expected, but necessary.
- **Claim (ii):** S3 (always-on overall centroid) and/or S4 (always-on mood centroid) score *worse than S2* on `factual` queries. **This is the surprising finding.** Quantify the drift — which entities get displaced when taste is applied inappropriately.
- **Claim (iii):** S5 matches or exceeds S2 on factual queries (router blocks taste) AND matches/exceeds S3+S4 on subjective queries (router still applies taste). S5 captures the upside without the downside.
- **Secondary:** Mood-specific centroids beat single-centroid on subjective queries; router accuracy ≥85%.
- **Efficiency:** Average latency under 3s, well below the ~12s a peer project hit with always-on LLM reranking.

**The 5-system design maps to claims directly:**
- S1 (plain LLM) → required baseline, proves retrieval helps at all
- S2 (no taste) → the reference everything else compares to
- S3 (always-on overall centroid) → tests claim (ii) — should hurt factual queries
- S4 (always-on mood centroid) → tests claim (ii) more aggressively — wrong-mood centroid on factual queries should hurt even more
- S5 (router + mood centroids) → tests claim (iii) — should win on both

---

## Architecture

```
        ┌──────────────┐
query → │ classify_    │ ── intent ∈ {factual, similarity, comparative,
        │  intent      │              mood_tragic, mood_epic, mood_political,
        │              │              mood_cathartic, mood_goofy, mood_general}
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │  retrieve    │ ── fusion score:
        │  (fusion)    │    α·query_sim + β·taste_align + γ·meta_match + δ·image_sim
        │              │    taste term gated by intent
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ synthesise   │ ── Gemini grounded answer from top-5 candidates
        └──────┬───────┘
               │
               ▼
            answer
```

**Modalities:** entity text (description + your_take + visual_description) + entity image. Two ChromaDB collections:
- `entities_text` — MiniLM (`all-MiniLM-L6-v2`) embeddings of combined text
- `entities_image` — CLIP (`clip-ViT-B-32`) embeddings of entity images

**Taste vectors:** weighted means of `your_take` embeddings (NOT the combined text — this keeps the personal signal pure). Weights = `(rating - 5) / 5`. Rated-10 entries contribute +1.0; rated-1 entries contribute -0.8 (centroid points away from things the user disliked). One overall centroid + 4 mood-specific sub-centroids. Sub-centroids only average entries tagged with that mood AND rated ≥7.

---

## The knowledge base — 70 entries

```
data/
├── entities.csv         # 70 rows × 14 columns
└── images/              # 70 .jpg files, named {entity_id}.jpg
```

### Schema (14 columns)

| Column | Purpose |
|---|---|
| `entity_id` | Stable ID (`c_*`, `s_*`, `p_*`, `e_*`) |
| `name` | Display name |
| `type` | character / ship / planet / episode |
| `era` | prequel / clone_wars / original / sequel / new_republic / legends |
| `faction` | Comma-separated, e.g. "Sith, Empire" |
| `first_appearance` | "A New Hope (1977)" |
| `medium` | film / show / book / game / comic (comma-separated) |
| `canon_status` | canon / legends / both |
| `description` | 1-2 sentences of canon facts (objective) |
| `your_take` | 1-2 sentences in user's voice — **drives the taste vector** |
| `your_rating` | 1-10 |
| `mood` | Comma-separated from: epic, tragic, political, cathartic, goofy |
| `visual_description` | 1-2 sentences purely visual — bridges text and CLIP |
| `image_note` | Which still/frame to grab (for reproducibility) |

### Composition

- **37 characters** including 4 droids (BB-8, R2-D2, C-3PO, K-2SO), the iconic villains (Vader, Palpatine ×2, Dooku, Maul, Grievous, Thrawn-Legends, Tarkin, Kylo, Phasma), heroes (Luke, Leia, Han, Chewbacca, two Obi-Wans, Yoda, Mace, Qui-Gon, Ahsoka, Anakin, Padmé, Cassian, Mon Mothma, Luthen, Bo-Katan, Din, Grogu), key supporting characters (Hux, Jar Jar, Rey, Revan-Legends), and the iconic faceless Stormtroopers
- **12 ships** (Falcon, X-wing, TIE, Star Destroyer, Death Star, U-wing, Razor Crest, Slave I, Naboo Starfighter, AT-AT, Jedi Interceptor, Republic Gunship)
- **11 planets** — biome-diverse: desert (Tatooine, Jedha), ocean-storm (Kamino), tropical beach (Scarif), ice (Hoth), swamp (Dagobah), forest (Endor), lava (Mustafar), city (Coruscant), salt-flat (Crait), Sith hellscape (Exegol)
- **10 episodes/arcs** (Order 66, Vader hallway, Obi-Wan/Vader rematch, Anakin's fall, Andor S1, Mando S2 finale, Book of Boba, Holiday Special, Kessel Run, full Kenobi show)

### Mood centroid health (entries rated ≥7 per mood)

- `tragic`: 24 ✅
- `epic`: 40 ✅
- `political`: 13 ✅
- `cathartic`: 9 ✅
- `goofy`: 8 ✅

**Six centroids total**: `overall` + 5 mood-specific. Every centroid has comfortably more than the minimum 5 contributors.

### Rating distribution

- 9-10: 37 entries (favourites — well-represented across moods)
- 6-8: 26 entries (the mid range)
- 2-5: 7 entries (dislikes — provides the negative pole for the overall centroid: Palpatine-sequels, Holiday Special, Phasma, Book of Boba, Exegol, Kylo, Rey)

### Rating distribution

- 9-10: 14 entries (favourites)
- 6-8: 28 entries (solid middle)
- 2-5: 15 entries (dislikes — give the taste vector negatives to point away from)

---

## Test set (`eval/test_set.json`) — 20 queries, 5 per family

1. **Factual** — title/field lookup. Router sends to factual path, taste OFF.
2. **Cross-modal** — visual anchors. Router sends to similarity, taste OFF.
3. **Multi-hop** — multi-attribute filters. Router sends to comparative, taste OFF.
4. **Ambiguous/personalised** — subjective queries. Router sends to `mood_*`, taste ON with appropriate centroid.

**Critical rule: never modify the test set after running an eval.** If you find a wrong gold label, document it in the report rather than rewrite history.

---

## Systems to compare (the eval is the marks)

| ID | System | What it tests |
|---|---|---|
| S1 | Plain Gemini, no retrieval | Required baseline. Fails on factual queries about your catalogue. |
| S2 | Hybrid retrieval, no taste, no agent | "Fixed pipeline" baseline. |
| S3 | Hybrid + single overall taste centroid (always on) | Does a single centroid help? Does it hurt factual queries? |
| S4 | Hybrid + mood centroids, no router (always applied) | Are sub-centroids worth it? Does always-on routing hurt? |
| S5 | **Full agent: router + mood centroids** | Proposed system. |

---

## Metrics reported

Per system, per family, and overall:
- **Recall@5** — primary retrieval quality
- **Hit@3** — strict precision
- **MRR** — ranking quality
- **Router accuracy** (S5 only) — does intent classification work?
- **LLM-as-judge groundedness** — Gemini scores each answer 1-5 for groundedness in retrieved metadata
- **Mean latency (ms)** — efficiency

---

## Key design decisions and why

| Decision | Why |
|---|---|
| Separate `description` and `your_take` fields | Lets the taste centroid be built purely from personal opinion text, not muddied by canon facts. |
| `visual_description` as third text field | Bridges text retrieval and CLIP — cross-modal queries hit both pathways. Concrete report ablation. |
| Mood sub-centroids (not single centroid) | Averaging across moods produces a blurry vector. Sub-centroids are sharper, give a clean secondary ablation. |
| Negative weighting in centroid | A vector pointing *away* from disliked entities is more informative than averaging favourites. |
| Router decides when taste fires | Applying taste to factual queries would *hurt*. This separation is the core of the hypothesis. |
| Two ChromaDB collections (not unified multimodal) | Lets us ablate text vs image cleanly. |
| Gemini 2.5 Flash | Paid tier, fast (~500ms), structured JSON output. |
| 4 LangGraph nodes max | Keep the graph small. The interesting thing is fusion + routing, not graph depth. |

---

## Failure modes to watch for

1. **Sloppy `your_take` fields kill the taste vector.** If they read like generic Wookieepedia summaries, the embeddings won't separate moods. The takes in `entities.csv` must remain in the user's voice.
2. **Test set leakage.** Writing queries after seeing what the system retrieves = overfitting. Test set is locked.
3. **Router classifies subjective queries as factual.** This kills the hypothesis. Report router accuracy separately.
4. **Gemini structured-output failures.** Always wrap classifier calls in JSON validation with fallback to a default intent.
5. **Image filenames not matching `entity_id`.** `ingest.py` looks for `{entity_id}.jpg`. Wrong filename = silent skip from the image collection.

---

## Where to start (if you're picking this up cold)

1. Read this file (done).
2. Look at `entities.csv` — the KB is already populated.
3. Drop 57 cover images into `data/images/`, named `{entity_id}.jpg`. See `IMAGE_DOWNLOAD_GUIDE.md`.
4. Run `python -m src.ingest` to build the ChromaDB collections.
5. Run `python -m src.taste` to compute centroids and inspect the PCA plot.
6. Run `python -m src.agent "a deeply tragic character arc"` to sanity check the agent.
7. Look at `eval/test_set.json` — locked.
8. Run `python -m eval.evaluate` to produce the comparison table.
9. Write the report.

---

## Hard rules

- **Never** modify `eval/test_set.json` after running an eval.
- **Never** train or fine-tune anything. The centroid is a NumPy mean, not a learned parameter.
- **Never** put more than 4 nodes in the LangGraph.
- **Always** keep the `use_taste` flag in retrieval so ablations are one line of code.
- **Always** record latency per retrieval call, even during dev.
