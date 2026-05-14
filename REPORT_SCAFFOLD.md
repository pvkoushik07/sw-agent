# Report Scaffold — 4-Page Systems Paper

> Use this as a writing skeleton. Each section has a target length, a "what goes here" note, and prompts for what to fill in once you have results.

---

## Section 0 — Title, abstract (≤ 100 words, on cover/header)

**Title suggestion:** *Selective Personalisation in Multimodal Retrieval: When User Preferences Help and When They Hurt*

**Abstract:** Personalisation is widely assumed to improve retrieval. We hypothesise that *uniform* personalisation actively degrades factual queries by pulling results toward the user's favourites, and that selective application — gated by query intent — captures personalisation's upside without the downside. Using a curated 57-entity Star Wars knowledge base, we compare five system variants: a plain LLM baseline, a hybrid retriever, two always-on personalisation variants, and a router-gated agent. Results show [FILL: e.g. always-on taste reduces factual Recall@5 by X%, while gated personalisation restores factual accuracy and improves subjective queries by Y%].

---

## Section 1 — Problem framing & hypothesis (≈ 0.6 page)

This is what the 4-marks "Problem Framing & Innovation" rubric grades. Make the framing sharp.

**Open with the puzzle:**
- Most personalised retrieval systems apply user signals uniformly to every query.
- This is intuitive: "we know what the user likes; bias results toward it."
- But: queries differ in subjectivity. "What did I rate Hades?" has a single correct answer. "Recommend a tragic arc" depends entirely on the user.
- Applying the same personalisation signal to both seems plausible but is **untested**.

**State the three claims explicitly:**

> **(i)** A personal taste vector improves subjective query performance.
>
> **(ii)** The same vector *degrades* factual query performance by pulling results toward favourites — uniform personalisation can hurt.
>
> **(iii)** An intent-gated router captures (i)'s upside while eliminating (ii)'s downside.

**Position vs. prior work:**
- Hybrid retrieval (your friend's project, teaching demo) explores the multimodal axis but applies retrieval uniformly.
- LLM rerankers improve quality at large latency cost (peer project: ~12s/query).
- Our contribution is orthogonal: *when* to personalise, not *how* to retrieve better.

---

## Section 2 — Knowledge base (≈ 0.5 page)

The 4-marks "Knowledge Base & Retrieval Design" rubric.

**What to write:**
- 57 personally curated Star Wars entities across 4 types (28 characters, 12 ships, 8 planets, 7 episodes/arcs)
- Coverage: canon film, shows, and 2 Legends entries (Thrawn, Revan)
- The schema separates `description` (canon facts) from `your_take` (personal opinion), enabling the taste vector to be built from personal text only, not muddied by canon
- Image collection: 57 entity images embedded via CLIP
- Personalisation is real: `your_take` fields are in the author's voice with specific opinions ("the Rogue One hallway scene is the single best minute of Star Wars on film"), not scraped descriptions

**Insert table 1:** Rating distribution + per-mood centroid counts (already computed by `taste.py`)

| Stat | Value |
|---|---|
| Total entries | 57 |
| Avg rating | [fill] |
| Entries rated ≥7 (taste-vector-eligible) | [fill] |
| Entries rated ≤4 (negative pole) | [fill] |
| Mood `tragic` centroid contributors | 21 |
| Mood `epic` centroid contributors | 33 |
| Mood `political` centroid contributors | 12 |
| Mood `cathartic` centroid contributors | 8 |

**Why personal reviews matter for the claim:** if the taste vector were built from generic Wookieepedia text, every user would get the same centroid — no personalisation. The personal reviews ensure the centroid is genuinely user-specific.

---

## Section 3 — Retrieval design (≈ 0.7 page)

**The fusion equation:**
$$\text{score}(q, e) = \alpha \cdot \text{sim}_{\text{text}}(q, e) + \beta \cdot \text{taste}(e, c) + \gamma \cdot \text{meta}(q, e) + \delta \cdot \text{sim}_{\text{img}}(q, e)$$

With $\alpha=0.50, \beta=0.30, \gamma=0.15, \delta=0.05$. The $\beta$ term is *gated* by query intent — set to 0 for factual/similarity/comparative queries.

**Two ChromaDB collections:** MiniLM for text (description + your_take + visual_description concatenated), CLIP for images. Separate to enable clean ablations.

**Taste centroid construction:**
- Embed `your_take` field *only* (not combined text) for purity
- Overall centroid: $c = \sum_i w_i x_i / \sum_i |w_i|$ where $w_i = (r_i - 5) / 5$
- Negative weights for low-rated entries push the centroid *away* from disliked entities — informative beyond a simple favourites-mean
- Mood sub-centroids: uniform mean over entries tagged with that mood AND rated ≥ 7
- Five centroids total: overall + tragic + epic + political + cathartic

**Insert figure 1 here:** The PCA plot from `taste.py` showing entities coloured by rating with all 5 centroids overlaid. **Visual evidence** that the centroids occupy distinct regions and that the overall centroid sits among high-rated entries.

**Why mood sub-centroids (not just overall):** averaging across moods produces a blurry vector. Sub-centroids are sharper because they aggregate over a coherent subset. This is testable: S3 (single centroid) vs S4 (mood centroids) on subjective queries.

---

## Section 4 — Agent workflow (≈ 0.5 page)

The 4-marks "Agent Framework & Tool Orchestration" rubric.

**4-node LangGraph:** classify → retrieve → synthesise → END.

**Insert figure 2 here:** Architecture diagram (use the ASCII one in `CLAUDE.md` or redraw).

**Classify node:** Gemini 2.5 Flash with structured JSON output. 8 intent labels: `factual`, `similarity`, `comparative`, `mood_{tragic,epic,political,cathartic,general}`. Each intent maps to a `(use_taste, taste_key)` pair.

**Retrieve node:** runs the fusion equation with `use_taste` and `taste_key` from the classifier.

**Synthesise node:** Gemini receives the top-5 candidates with full metadata and writes a 2-4 sentence grounded answer, quoting the user's own `your_take` field where relevant. Prompt explicitly forbids hallucinating entities not in the candidate set.

**Why the router design matters for the hypothesis:** Without the router, taste is either always on (S3/S4) or always off (S2). Only the router lets us test claim (iii) — that selective application captures both benefits.

**Trace example (insert one real trace from a debug run):**
```
Query: "a deeply tragic character arc"
  classify → intent=mood_tragic, conf=0.92, use_taste=True, taste_key=mood_tragic
  retrieve → top-5: [c_anakin, c_cassian, e_order66, c_obiwan_pt, c_ahsoka]
  synthesise → "Anakin Skywalker is the obvious recommendation — you called..."
  trace: classify=520ms, retrieve=180ms, synthesise=780ms, total=1480ms
```

---

## Section 5 — Experiments & ablation (≈ 1.0 page — the biggest section)

The 4-marks "Quantitative Evaluation & Ablation" rubric. This is where the marks live.

**Test set:** 20 hand-labelled queries, 5 per family (factual / cross-modal / multi-hop / ambiguous_personalised), with 1-5 gold entity IDs per query. Test set locked before retrieval weight tuning (no overfitting).

**Five systems compared:**

| ID | Variant | Tests |
|---|---|---|
| S1 | Plain Gemini, no retrieval | Required baseline |
| S2 | Hybrid retrieval, no taste | Reference point |
| S3 | Hybrid + always-on overall centroid | Tests claim (ii) — should hurt factual |
| S4 | Hybrid + always-on mood centroid (mood_tragic default) | Tests claim (ii) more aggressively |
| S5 | Router + mood centroids (proposed) | Tests claim (iii) |

**Metrics:** Recall@5, Hit@3, MRR (retrieval); LLM-as-judge groundedness 1-5 (answer quality); mean latency (efficiency); router accuracy (S5 only).

**Insert table 2 here:** Overall results from `eval/results/summary_overall.csv`.

**Insert table 3 here:** Per-family breakdown from `eval/results/summary_by_family.csv`. This is the report's centrepiece — read down the `factual` column for claim (ii), down the `ambiguous_personalised` column for claim (i).

**Claim (ii) deep-dive — the surprising finding:**

Insert table 4 here: drift analysis from `eval/results/drift_summary.csv`. Read off: "On factual queries, S3 displaced X gold entities from top-5 vs S2 baseline. S4 displaced Y. S5 displaced Z (~0)."

[FILL after running eval: discuss *which specific entities* got displaced. Probably the user's favourites (Vader, Han, Andor S1) crowding out the correct factual answers. Quote 1-2 specific examples.]

**Claim (iii) discussion:**
[FILL: compare S5 vs S3 on factual (router protects) and vs S2 on subjective (router still enables taste).]

**Router accuracy:** [FILL: X/20 correct]. Discuss the failure cases — which queries did the router misclassify, and did the misclassification hurt retrieval?

**Latency comparison:**
[FILL: S5 average latency, contrast with peer project's ~12s figure.]

---

## Section 6 — Failure analysis (≈ 0.3 page)

Pick 2-3 queries where the system loses. Examples likely to fail:
- **A5 (Surprise me with something I love that I don't talk about much):** fuzzy gold, overall centroid may pull toward most-talked-about favourites instead of overlooked ones
- **C5 (Small green creature with long ears):** does the system surface BOTH Yoda and Grogu, or only one?
- **Any factual query the router misclassified:** quote the query, the wrong intent, and the consequence

For each: explain what went wrong, why, and what would fix it (without claiming you'd implement it).

---

## Section 7 — Discussion & limitations (≈ 0.4 page)

**Key findings restated:**
1. Personalisation helps subjective queries (claim i confirmed).
2. Personalisation hurts factual queries when applied uniformly (claim ii — the surprising one).
3. Intent gating captures both benefits (claim iii confirmed).

**Generality argument:** the "selective application" principle isn't Star Wars-specific. It applies to any personalised retrieval system where queries vary in subjectivity (recipes, films, books, papers).

**Limitations:**
- Single user — the personalisation observation may differ for users with broader/narrower tastes
- 57 entities is small — at production scale, drift effects may be different
- Test set is hand-labelled by the same user whose taste drives the system; potential confound, though we mitigate by writing the test set before tuning

**Future work (one sentence):** comparing gated personalisation against an LLM reranker baseline (cost / quality trade-off) is a natural next experiment.

---

## Final checklist before submission

- [ ] Hypothesis stated clearly at the top — all three claims (i), (ii), (iii)
- [ ] Table 1: KB stats
- [ ] Figure 1: PCA plot of taste centroids
- [ ] Figure 2: Architecture diagram
- [ ] Table 2: Overall results
- [ ] Table 3: Per-family breakdown
- [ ] Table 4: Drift analysis
- [ ] At least one trace example
- [ ] 2-3 failure cases honestly discussed
- [ ] Page count ≤ 4 (excluding appendix)
- [ ] Code, test set, and results CSVs all in the submitted zip
