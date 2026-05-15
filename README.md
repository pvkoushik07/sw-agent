# Selective Personalisation in a Multimodal Star Wars Retrieval Agent

This project implements a personalised multimodal retrieval agent over a
70-entity Star Wars catalogue (characters, ships, planets, and episodes/arcs).
The agent fuses text similarity, image similarity (CLIP), metadata keyword
matching, and a personal taste vector derived from the author's own ratings and
written takes. The central hypothesis has three claims: (i) a personal taste
vector improves retrieval on subjective queries; (ii) the same vector degrades
retrieval on factual queries by pulling results toward the user's favourites;
(iii) an intent-classifying router that applies taste only on subjective queries
captures the upside of (i) while eliminating the downside of (ii). The headline
result: S5 (router + mood centroids) matches S2 (no taste) exactly on factual
queries with zero gold-entity drift, while beating it by 20pp Hit@3 on subjective
queries. A modality ablation (S6/S7/S8) shows the text channel is load-bearing
-- removing it drops factual Recall@5 by 38pp -- while CLIP earns its 0.05 weight
through improved ranking on cross-modal queries (MRR +10pp). A conversational
memory extension adds a resolve_references node that rewrites referential queries
using the previous turn's retrieved entities, producing +25pp Recall@5 on
follow-up queries compared to running without history.

## Author

Koushik Patnam Venkata, Student ID 48595843, INFS4205/7205 Assignment 3.

## Requirements

- Python 3.11+
- A Google API key for Gemini 2.5 Flash (set as `GEMINI_API_KEY` in `.env`)
- Approximately 2 GB disk for ChromaDB collections and image embeddings
- First run downloads MiniLM (`all-MiniLM-L6-v2`) and CLIP (`clip-ViT-B-32`)
  model weights, roughly 500 MB total

Package dependencies are listed in `requirements.txt`.

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your Gemini API key
cp .env.example .env
# Edit .env and set: GEMINI_API_KEY=<your key>

# 4. Build ChromaDB collections and taste centroids (~3-5 minutes on first run)
python -m src.ingest
python -m src.taste

# 5. Launch the interactive chat app
streamlit run src/app.py
```

Step 4 encodes all 70 entities with MiniLM (text) and CLIP (images), stores
them in two ChromaDB collections, then computes six taste centroids (one overall
plus five mood sub-centroids) from the `your_take` embeddings weighted by rating.

## Reproducing the evaluation

```bash
# Main evaluation: S1-S5 on the 20-query test set (~15-25 minutes)
python -m eval.evaluate

# Modality ablation: S6 text-only, S7 image-only, S8 sanity check
python -m eval.evaluate_modality

# Conversational memory ablation: FU1-FU4 with vs without resolve_references
python -m eval.evaluate_memory

# Generate all figures (fig3 through fig6 + taste_pca)
python -m eval.make_plots
```

`eval.evaluate` makes approximately 220 Gemini API calls.
`eval.evaluate_modality` and `eval.evaluate_memory` together add roughly
15-20 minutes of additional Gemini API time. If a run is interrupted, delete
the corresponding raw CSV and re-run from scratch.

Pre-computed results are included in `eval/results/` so re-running is optional.

## Project structure

```
sw_agent/
├── src/
│   ├── config.py              # All paths, model names, fusion weights, intent labels
│   ├── ingest.py              # Builds entities_text and entities_image ChromaDB collections
│   ├── taste.py               # Computes taste centroids, writes .chroma/taste_centroids.npz
│   ├── retrieve.py            # Fusion retrieval: text + taste + meta + image
│   ├── agent.py               # LangGraph graph: classify -> resolve_references -> retrieve -> synthesise
│   ├── llm.py                 # Gemini wrappers: classify_intent, synthesise_answer,
│   │                          #   resolve_query_reference, judge_groundedness
│   └── app.py                 # Streamlit UI: Chat (with history), Debug, Catalogue tabs
├── eval/
│   ├── test_set.json          # 24 locked queries across 5 families (incl. 4 conversational)
│   ├── evaluate.py            # Runs S1-S5 on the original 20-query test set
│   ├── evaluate_modality.py   # Modality ablation: S6/S7/S8 with try/finally weight patching
│   ├── evaluate_memory.py     # Memory ablation: FU1-FU4 with vs without conversation history
│   └── make_plots.py          # Generates fig3, fig4, fig5, fig6 and taste_pca.png
├── data/
│   ├── entities.csv           # 70-entity knowledge base (14 columns incl. your_take, your_rating)
│   └── images/                # 70 entity cover images (.jpg), named {entity_id}.jpg
├── requirements.txt
├── .env.example
└── README.md
```

## Design overview

**Knowledge base.** The catalogue contains 70 Star Wars entities across four
types: characters, ships, planets, and episodes/arcs. Each entry has a
`description` field (objective canon facts) and a separate `your_take` field
(the author's opinion, written in first person). Ratings are 1-10. Entries also
carry `mood` tags (epic, tragic, political, cathartic, goofy) and a
`visual_description` field that bridges text retrieval and CLIP for cross-modal
queries.

**Retrieval.** Each query produces a fusion score:

```
score(q, e) = 0.50 * text_sim(q, e)
            + 0.30 * taste_align(e, centroid)   # zero when use_taste=False
            + 0.15 * meta_match(q, e)
            + 0.05 * image_sim(q, e)
```

Text embeddings use `all-MiniLM-L6-v2`; image embeddings use `clip-ViT-B-32`.
The taste centroid is a rating-weighted mean of `your_take` embeddings only
(not the combined text), with weights `(rating - 5) / 5` so that low-rated
entries contribute negatively. Six centroids are computed: one overall and five
mood sub-centroids (epic, tragic, political, cathartic, goofy). The taste term
is gated by query intent and set to zero for factual, similarity, and
comparative queries.

**Agent.** The LangGraph pipeline has four nodes. The classify node sends the
query to Gemini 2.5 Flash with a structured JSON prompt and returns one of nine
intent labels. The resolve_references node inspects the query for referential
phrases ("those", "of those", "which of them", etc.) and, when history is
non-empty, calls Gemini to rewrite the query into a self-contained form
incorporating the previous turn's retrieved entity names. Agent state carries
the last five conversation turns; history is passed in by the caller and updated
after each run. The retrieve node calls the fusion function with the use_taste
and taste_key values set by the classifier. The synthesise node sends the top-5
candidates to Gemini with a grounding prompt that forbids hallucinating entities
not in the retrieved set.

## Systems evaluated

**Main ablation (S1-S5) — tests the selective personalisation hypothesis:**

| ID | System | Description |
|----|--------|-------------|
| S1 | Plain LLM | Gemini only, no catalogue retrieval |
| S2 | Hybrid, no taste | Text + image + meta retrieval, taste vector off |
| S3 | Always-on overall centroid | Overall taste centroid applied to every query |
| S4 | Always-on mood_tragic centroid | mood_tragic centroid applied to every query |
| S5 | Full agent (proposed) | Router classifies intent, applies the right mood centroid only on subjective queries. Also runs resolve_references for conversational follow-ups. |

**Modality ablation (S6-S8) — tests which retrieval channel earns its weight:**

| ID | System | Description |
|----|--------|-------------|
| S6 | Text only | DELTA=0 (image channel silenced); all else as S5 |
| S7 | Image only | ALPHA=0, GAMMA=0 (text + meta silenced); all else as S5 |
| S8 | Full sanity | Default weights; result should match S5 within stochasticity |

## Files generated by the evaluation

| File | Description |
|------|-------------|
| `eval/results/raw_results.csv` | Per-query, per-system results with all metrics (S1-S5) |
| `eval/results/summary_overall.csv` | Per-system averages across all 20 original queries |
| `eval/results/summary_by_family.csv` | Per-system averages broken down by query family |
| `eval/results/drift_analysis.csv` | Per-query entity displacement vs S2 baseline |
| `eval/results/drift_summary.csv` | Drift aggregated by system and query family |
| `eval/results/modality_summary_overall.csv` | S6/S7/S8 averages across all 20 queries |
| `eval/results/modality_summary_by_family.csv` | S6/S7/S8 averages broken down by query family |
| `eval/results/memory_ablation.csv` | Turn-2 metrics for FU1-FU4 with vs without history |
| `eval/results/conversational_followup_summary.csv` | Aggregated memory ablation by mode |
| `eval/results/taste_pca.png` | PCA of your_take embeddings with taste centroids overlaid |
| `eval/results/fig3_recall_by_family.png` | Recall@5 per system per query family (S1-S5) |
| `eval/results/fig4_drift_factual.png` | Gold-entity drift on factual queries |
| `eval/results/fig5_latency.png` | End-to-end latency per system |
| `eval/results/fig6_modality_ablation.png` | Recall@5 by channel (S6/S7/S8) per family |

## Configuration

All tunable values are in `src/config.py`: fusion weights (ALPHA, BETA, GAMMA,
DELTA), model names, ChromaDB collection names, retrieval depth (TOP_K_FINAL),
taste centroid parameters (RATING_NEUTRAL, RATING_SCALE, MIN_RATING_FOR_MOOD),
and intent label vocabulary. Do not hard-code any of these in other modules.

The modality ablation scripts patch config values at runtime using a
`contextlib.contextmanager` that restores originals in `finally`, so running
`evaluate_modality.py` does not permanently alter any weights.

## Troubleshooting

- **Gemini quota errors during eval.** Delete the corresponding raw CSV and
  re-run. All three eval scripts are safe to restart from scratch.

- **ChromaDB collection already exists error.** Delete the `.chroma/` directory
  and re-run `python -m src.ingest`. The ingest script drops and rebuilds
  collections on each run, but a corrupt state can prevent this.

- **ModuleNotFoundError: No module named 'src'.** Confirm the virtual
  environment is activated (`source .venv/bin/activate`) and that
  `pip install -r requirements.txt` completed without errors. Run all commands
  from the `sw_agent/` root directory.

- **Memory ablation results look inconsistent across runs.** The
  resolve_references node calls Gemini to detect and rewrite referential
  queries. This introduces stochasticity: borderline cases (e.g. "which one"
  without "those") may or may not trigger rewriting depending on the model
  response. Results in `eval/results/memory_ablation.csv` reflect one
  deterministic run.

## Acknowledgement of AI use

This submission was built with assistance from Claude (Anthropic) for code
scaffolding (LangGraph agent skeleton, evaluation harness boilerplate) and for
prose structuring in the report. Ideation, system design, knowledge base
curation, personal ratings and takes, and the quantitative analysis are the
author's own.
