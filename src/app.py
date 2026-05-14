"""Streamlit UI. Run: streamlit run src/app.py"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src import ...` works regardless
# of where streamlit is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from src import config

st.set_page_config(
    page_title="Taste-Aware Star Wars Agent",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .intent-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-factual    { background:#1e3a5f; color:#7ec8e3; }
    .badge-similarity { background:#1e3a5f; color:#7ec8e3; }
    .badge-comparative{ background:#1e3a5f; color:#7ec8e3; }
    .badge-mood       { background:#3a1e5f; color:#c8a7e3; }
    .badge-taste-on   { background:#1e5f2a; color:#7ee39a; }
    .badge-taste-off  { background:#3a3a3a; color:#aaaaaa; }
    .answer-box {
        background: #0e1117;
        border-left: 3px solid #ffd700;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0 18px 0;
        font-size: 1.02rem;
        line-height: 1.6;
    }
    .entity-card {
        background: #1a1d23;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        height: 100%;
    }
    .score-bar-label {
        font-size: 0.72rem;
        color: #aaa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ─────────────────────────────────────────────────────────────────────

INTENT_EMOJI = {
    "factual": "🔍",
    "similarity": "👁️",
    "comparative": "⚖️",
    "mood_tragic": "💔",
    "mood_epic": "⚡",
    "mood_political": "🏛️",
    "mood_cathartic": "✨",
    "mood_goofy": "😄",
    "mood_general": "🌌",
}

INTENT_CLASS = {
    "factual": "badge-factual",
    "similarity": "badge-similarity",
    "comparative": "badge-comparative",
}


def intent_badge(intent: str, confidence: float) -> str:
    css = INTENT_CLASS.get(intent, "badge-mood")
    emoji = INTENT_EMOJI.get(intent, "🌌")
    return (
        f'<span class="intent-badge {css}">'
        f'{emoji} {intent} ({confidence:.0%})'
        f"</span>"
    )


def taste_badge(use_taste: bool, taste_key: str | None) -> str:
    if use_taste:
        return (
            f'<span class="intent-badge badge-taste-on">'
            f"✅ taste ON · {taste_key}"
            f"</span>"
        )
    return '<span class="intent-badge badge-taste-off">⬜ taste OFF</span>'


def _entity_image(entity_id: str):
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = config.IMAGES_DIR / f"{entity_id}{ext}"
        if p.exists():
            return str(p)
    return None


@st.cache_resource(show_spinner="Loading agent…")
def get_agent():
    from src.agent import run_agent  # noqa: deferred to avoid import at startup
    return run_agent


@st.cache_data(show_spinner=False)
def load_catalogue() -> pd.DataFrame:
    return pd.read_csv(config.ENTITIES_CSV)


# ── Render result cards ─────────────────────────────────────────────────────────

def render_results(results, debug: bool = False):
    if not results:
        st.warning("No results returned.")
        return

    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        meta = r.metadata
        img = _entity_image(meta["entity_id"])
        with col:
            if img:
                st.image(img, use_container_width=True)
            st.markdown(f"**{meta['name']}**")
            st.caption(
                f"⭐ {meta['your_rating']}/10 · {meta['type']} · {meta['era']}"
            )
            st.caption(f"*{meta['mood']}*")
            with st.expander("Your take"):
                st.write(meta["your_take"])
            if debug:
                total = r.final_score
                comps = r.components
                st.caption("Score components")
                for label, val in [
                    ("query_sim", comps["query_sim"]),
                    ("taste_align", comps["taste_align"]),
                    ("meta_score", comps["meta_score"]),
                    ("image_sim", comps["image_sim"]),
                ]:
                    st.progress(
                        min(float(val), 1.0),
                        text=f"{label}: {val:.3f}",
                    )
                st.caption(f"**final: {total:.3f}**")


# ── Tabs ────────────────────────────────────────────────────────────────────────

st.title("🌌 Taste-Aware Star Wars Agent")
st.caption("UQ INFS4205/7205 — A3 · Personalised Multimodal Retrieval")

tab_chat, tab_debug, tab_catalogue = st.tabs(["💬 Chat", "🔧 Debug", "📚 Catalogue"])


# ── TAB 1: Chat ─────────────────────────────────────────────────────────────────

with tab_chat:
    st.markdown("Ask anything about your Star Wars catalogue.")

    EXAMPLES = [
        "a deeply tragic character arc",
        "What did I rate Andor Season 1?",
        "The droid that looks like a soccer ball",
        "something cathartic that pays off a long setup",
        "Jedi I rated 9+ from the prequel era",
        "scheming political villains",
    ]

    st.markdown("**Try an example:**")
    ex_cols = st.columns(len(EXAMPLES))
    clicked = None
    for col, ex in zip(ex_cols, EXAMPLES):
        if col.button(ex, key=f"ex_{ex}", use_container_width=True):
            clicked = ex

    query = st.text_input(
        "Or type your own query",
        value=clicked or "",
        placeholder="a sweeping epic moment",
        key="q_chat",
    )

    if query:
        run_agent = get_agent()
        with st.spinner("Routing query and retrieving…"):
            out = run_agent(query)

        # Intent + taste routing row
        st.markdown(
            intent_badge(out.get("intent", "?"), out.get("intent_confidence", 0))
            + "  "
            + taste_badge(out.get("use_taste", False), out.get("taste_key")),
            unsafe_allow_html=True,
        )

        # Answer
        st.markdown(
            f'<div class="answer-box">{out["answer"]}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Top matches**")
        render_results(out.get("results", []))


# ── TAB 2: Debug ────────────────────────────────────────────────────────────────

with tab_debug:
    st.markdown("Same pipeline as Chat — shows full routing trace and score breakdown.")

    query_d = st.text_input(
        "Query",
        placeholder="Jedi I rated 9+ from the prequel era",
        key="q_debug",
    )

    if query_d:
        run_agent = get_agent()
        with st.spinner("Running…"):
            out = run_agent(query_d)

        trace = out.get("trace", {})

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Intent", f"{INTENT_EMOJI.get(out.get('intent','?'), '?')} {out.get('intent','?')}")
        m2.metric("Confidence", f"{out.get('intent_confidence', 0):.0%}")
        m3.metric("Taste", f"{'ON · ' + out.get('taste_key','') if out.get('use_taste') else 'OFF'}")
        m4.metric("Total latency", f"{trace.get('total_ms', 0):.0f} ms")

        # Timing breakdown
        st.markdown("**Node timings**")
        t_cols = st.columns(3)
        t_cols[0].metric("Classify", f"{trace.get('classify_ms', 0):.0f} ms")
        t_cols[1].metric("Retrieve", f"{trace.get('retrieve_ms', 0):.0f} ms")
        t_cols[2].metric("Synthesise", f"{trace.get('synthesise_ms', 0):.0f} ms")

        # Answer
        st.markdown("**Answer**")
        st.markdown(
            f'<div class="answer-box">{out["answer"]}</div>',
            unsafe_allow_html=True,
        )

        # Results with score breakdown
        st.markdown("**Retrieved results (with score components)**")
        render_results(out.get("results", []), debug=True)


# ── TAB 3: Catalogue ────────────────────────────────────────────────────────────

with tab_catalogue:
    st.markdown("Browse and filter your 70-entity Star Wars knowledge base.")

    df = load_catalogue()

    # Filters
    f1, f2, f3, f4 = st.columns(4)

    types = ["All"] + sorted(df["type"].unique().tolist())
    sel_type = f1.selectbox("Type", types)

    eras = ["All"] + sorted(df["era"].unique().tolist())
    sel_era = f2.selectbox("Era", eras)

    all_moods = sorted({
        m.strip()
        for moods in df["mood"].dropna()
        for m in moods.split(",")
    })
    sel_mood = f3.selectbox("Mood", ["All"] + all_moods)

    min_rating, max_rating = f4.slider("Rating range", 1, 10, (1, 10))

    # Apply filters
    mask = (df["your_rating"] >= min_rating) & (df["your_rating"] <= max_rating)
    if sel_type != "All":
        mask &= df["type"] == sel_type
    if sel_era != "All":
        mask &= df["era"] == sel_era
    if sel_mood != "All":
        mask &= df["mood"].str.contains(sel_mood, case=False, na=False)

    filtered = df[mask].sort_values("your_rating", ascending=False).reset_index(drop=True)
    st.caption(f"{len(filtered)} of {len(df)} entities")

    # Display as cards (5 per row)
    CARDS_PER_ROW = 5
    for row_start in range(0, len(filtered), CARDS_PER_ROW):
        row_df = filtered.iloc[row_start : row_start + CARDS_PER_ROW]
        cols = st.columns(CARDS_PER_ROW)
        for col, (_, row) in zip(cols, row_df.iterrows()):
            with col:
                img = _entity_image(row["entity_id"])
                if img:
                    st.image(img, use_container_width=True)
                st.markdown(f"**{row['name']}**")
                st.caption(f"⭐ {row['your_rating']}/10 · {row['type']}")
                st.caption(f"{row['era']} · *{row['mood']}*")
                with st.expander("Your take"):
                    st.write(row["your_take"])
