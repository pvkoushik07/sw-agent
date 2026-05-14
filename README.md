# Taste-Aware Star Wars Universe Agent

A LangGraph agent that retrieves Star Wars entities from a personally curated catalogue, using a learned **taste vector** to handle subjective queries like "a deeply tragic character arc" or "something cathartic that pays off a long setup."

UQ INFS4205/7205 — Assignment 3.

> For project context, design rationale, and hard rules, see [`CLAUDE.md`](./CLAUDE.md).
> For how to download the 57 images, see [`IMAGE_DOWNLOAD_GUIDE.md`](./IMAGE_DOWNLOAD_GUIDE.md).

---

## Setup

```bash
# 1. Enter the project
cd sw_agent

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your Gemini API key
cp .env.example .env
# Edit .env and add: GEMINI_API_KEY=...
```

---

## Run

```bash
# Step 1: Download the 57 images into data/images/ (see IMAGE_DOWNLOAD_GUIDE.md)

# Step 2: Build the ChromaDB collections
python -m src.ingest

# Step 3: Compute taste centroids (writes eval/results/taste_pca.png)
python -m src.taste

# Step 4: Sanity check the agent
python -m src.agent "a deeply tragic character arc"

# Step 5: Launch the Streamlit UI
streamlit run src/app.py

# Step 6: Run the full evaluation across 5 system variants
python -m eval.evaluate
```

---

## Status / TODOs

- [x] Knowledge base: 70 entries written into `data/entities.csv`
- [x] Test set: 20 queries with gold IDs in `eval/test_set.json`
- [ ] **Download 70 images into `data/images/`** ← your job, see guide
- [ ] Run ingestion
- [ ] Run evaluation across 5 system variants
- [ ] Write the 4-page report
