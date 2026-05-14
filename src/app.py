"""Streamlit UI. Run: streamlit run src/app.py"""
from __future__ import annotations
import sys
import time
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

# ── Custom CSS ──────────────────────────────────────────────────────────────────
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
    .sys-S1 { background:#2a2a2a; color:#aaaaaa; }
    .sys-S2 { background:#1e3a5f; color:#7ec8e3; }
    .sys-S3 { background:#5f3a1e; color:#e3b07e; }
    .sys-S4 { background:#5f1e1e; color:#e37e7e; }
    .sys-S5 { background:#4a3a00; color:#ffd700; }
    .answer-box {
        background: #0e1117;
        border-left: 3px solid #ffd700;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0 18px 0;
        font-size: 1.02rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── System definitions ──────────────────────────────────────────────────────────

SYSTEM_META = {
    "S1 — Plain LLM (no retrieval)": {
        "id": "S1", "short": "S1: Plain LLM",
        "desc": "Gemini only, no catalogue access. Always wrong on factual queries.",
        "use_taste": False, "taste_key": None,
    },
    "S2 — Hybrid retrieval, no taste": {
        "id": "S2", "short": "S2: No taste",
        "desc": "Text + image retrieval. Taste vector never applied.",
        "use_taste": False, "taste_key": None,
    },
    "S3 — Always-on overall centroid": {
        "id": "S3", "short": "S3: Always overall taste",
        "desc": "Overall taste centroid always applied — even on factual queries.",
        "use_taste": True, "taste_key": "overall",
    },
    "S4 — Always-on mood_tragic centroid": {
        "id": "S4", "short": "S4: Always mood_tragic",
        "desc": "mood_tragic centroid always applied — wrong centroid on non-tragic queries.",
        "use_taste": True, "taste_key": "mood_tragic",
    },
    "S5 — Full agent: router + mood centroids": {
        "id": "S5", "short": "S5: Full agent ✦",
        "desc": "Router classifies intent, applies the right mood centroid only when helpful.",
        "use_taste": None, "taste_key": None,  # determined at runtime
    },
}

SYSTEM_LABELS = list(SYSTEM_META.keys())

# ── Helpers ─────────────────────────────────────────────────────────────────────

INTENT_EMOJI = {
    "factual": "🔍", "similarity": "👁️", "comparative": "⚖️",
    "mood_tragic": "💔", "mood_epic": "⚡", "mood_political": "🏛️",
    "mood_cathartic": "✨", "mood_goofy": "😄", "mood_general": "🌌",
}

INTENT_CSS = {
    "factual": "badge-factual", "similarity": "badge-similarity",
    "comparative": "badge-comparative",
}


def system_badge(sys_id: str, short: str) -> str:
    return f'<span class="intent-badge sys-{sys_id}">{short}</span>'


def intent_badge(intent: str, confidence: float) -> str:
    css = INTENT_CSS.get(intent, "badge-mood")
    emoji = INTENT_EMOJI.get(intent, "🌌")
    return f'<span class="intent-badge {css}">{emoji} {intent} ({confidence:.0%})</span>'


def taste_badge(use_taste: bool, taste_key: str | None) -> str:
    if use_taste:
        return f'<span class="intent-badge badge-taste-on">✅ taste ON · {taste_key}</span>'
    return '<span class="intent-badge badge-taste-off">⬜ taste OFF</span>'


def _entity_image(entity_id: str) -> str | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = config.IMAGES_DIR / f"{entity_id}{ext}"
        if p.exists():
            return str(p)
    return None


@st.cache_resource(show_spinner="Loading models…")
def _load_retrieve_and_llm():
    from src.retrieve import retrieve
    from src import llm
    return retrieve, llm


@st.cache_resource(show_spinner="Loading agent…")
def _load_agent():
    from src.agent import run_agent
    return run_agent


def _init_session():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "chat_turns" not in st.session_state:
        st.session_state.chat_turns = []  # [{query, answer, intent, use_taste, taste_key, system}]


@st.cache_data(show_spinner=False)
def load_catalogue() -> pd.DataFrame:
    return pd.read_csv(config.ENTITIES_CSV)


# ── Run a system variant ────────────────────────────────────────────────────────

def run_system(query: str, label: str, history: list[dict] | None = None) -> dict:
    meta = SYSTEM_META[label]
    sys_id = meta["id"]

    if sys_id == "S1":
        t0 = time.perf_counter()
        answer = "I don't have access to your catalogue in this mode — this is the plain LLM baseline."
        return {
            "system_id": sys_id, "system_short": meta["short"],
            "answer": answer, "results": [],
            "use_taste": False, "taste_key": None,
            "intent": None, "intent_confidence": None,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "reference_detected": False, "resolved_query": None,
            "updated_history": history or [],
        }

    if sys_id == "S5":
        run_agent = _load_agent()
        out = run_agent(query, history=history or [])
        results = out.get("results", [])
        return {
            "system_id": sys_id, "system_short": meta["short"],
            "answer": out.get("answer", ""),
            "results": results,
            "use_taste": out.get("use_taste", False),
            "taste_key": out.get("taste_key"),
            "intent": out.get("intent"),
            "intent_confidence": out.get("intent_confidence"),
            "latency_ms": out.get("trace", {}).get("total_ms", 0),
            "reference_detected": out.get("reference_detected", False),
            "resolved_query": out.get("resolved_query"),
            "updated_history": out.get("history", []),
        }

    # S2 / S3 / S4 — no router, no memory
    retrieve, llm = _load_retrieve_and_llm()
    t0 = time.perf_counter()
    trace = retrieve(query, use_taste=meta["use_taste"], taste_key=meta["taste_key"] or "overall")
    synth = llm.synthesise_answer(query, [r.metadata for r in trace.results])
    return {
        "system_id": sys_id, "system_short": meta["short"],
        "answer": synth,
        "results": trace.results,
        "use_taste": meta["use_taste"],
        "taste_key": meta["taste_key"],
        "intent": None,
        "intent_confidence": None,
        "latency_ms": (time.perf_counter() - t0) * 1000,
        "reference_detected": False, "resolved_query": None,
        "updated_history": [],
    }


# ── Render result cards ─────────────────────────────────────────────────────────

def render_results(results, debug: bool = False):
    if not results:
        st.info("No retrieved results for this system.")
        return
    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        meta = r.metadata
        img = _entity_image(meta["entity_id"])
        with col:
            if img:
                st.image(img, use_container_width=True)
            st.markdown(f"**{meta['name']}**")
            st.caption(f"⭐ {meta['your_rating']}/10 · {meta['type']} · {meta['era']}")
            st.caption(f"*{meta['mood']}*")
            with st.expander("Your take"):
                st.write(meta["your_take"])
            if debug:
                comps = r.components
                for lbl, val in [
                    ("query_sim", comps["query_sim"]),
                    ("taste_align", comps["taste_align"]),
                    ("meta_score", comps["meta_score"]),
                    ("image_sim", comps["image_sim"]),
                ]:
                    st.progress(min(float(val), 1.0), text=f"{lbl}: {val:.3f}")
                st.caption(f"**final: {r.final_score:.3f}**")


# ── Layout ──────────────────────────────────────────────────────────────────────

_init_session()

st.title("🌌 Taste-Aware Star Wars Agent")
st.caption("UQ INFS4205/7205 — A3 · Personalised Multimodal Retrieval")

tab_chat, tab_debug, tab_catalogue = st.tabs(["💬 Chat", "🔧 Debug", "📚 Catalogue"])


# ── TAB 1: Chat ─────────────────────────────────────────────────────────────────

with tab_chat:
    # System selector + clear button on one row
    ctrl_left, ctrl_right = st.columns([5, 1])
    with ctrl_left:
        sel_system = st.radio(
            "system",
            SYSTEM_LABELS,
            index=4,
            horizontal=True,
            label_visibility="collapsed",
        )
    with ctrl_right:
        if st.button("🗑 Clear", use_container_width=True, help="Clear conversation history"):
            st.session_state.history = []
            st.session_state.chat_turns = []
            st.rerun()

    st.caption(f"_{SYSTEM_META[sel_system]['desc']}_")
    if sel_system != "S5 — Full agent: router + mood centroids":
        st.caption("_Note: conversation memory only works with S5 (router resolves references)._")

    st.divider()

    # Conversation history display
    if st.session_state.chat_turns:
        st.markdown("**Conversation so far**")
        for turn in st.session_state.chat_turns:
            with st.chat_message("user"):
                st.write(turn["query"])
            with st.chat_message("assistant"):
                if turn.get("reference_detected"):
                    st.caption(f"_Reference resolved: \"{turn['resolved_query']}\"_")
                badges = system_badge(turn["system_id"], turn["system_short"]) + "  "
                if turn.get("intent"):
                    badges += intent_badge(turn["intent"], turn.get("intent_confidence") or 0) + "  "
                badges += taste_badge(turn.get("use_taste", False), turn.get("taste_key"))
                st.markdown(badges, unsafe_allow_html=True)
                st.markdown(
                    f'<div class="answer-box">{turn["answer"]}</div>',
                    unsafe_allow_html=True,
                )
        st.divider()

    # Example queries
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
        "Or type your query",
        value=clicked or "",
        placeholder="a sweeping epic moment",
        key="q_chat",
    )

    if query:
        history_for_s5 = st.session_state.history if sel_system == "S5 — Full agent: router + mood centroids" else []
        with st.spinner(f"Running {SYSTEM_META[sel_system]['short']}…"):
            out = run_system(query, sel_system, history=history_for_s5)

        # Update session history (S5 only)
        if sel_system == "S5 — Full agent: router + mood centroids":
            st.session_state.history = out.get("updated_history", [])

        # Store turn for history display
        st.session_state.chat_turns.append({
            "query": query,
            "answer": out["answer"],
            "system_id": out["system_id"],
            "system_short": out["system_short"],
            "intent": out.get("intent"),
            "intent_confidence": out.get("intent_confidence"),
            "use_taste": out.get("use_taste", False),
            "taste_key": out.get("taste_key"),
            "reference_detected": out.get("reference_detected", False),
            "resolved_query": out.get("resolved_query"),
        })

        # Current turn badges
        badges = system_badge(out["system_id"], out["system_short"]) + "  "
        if out.get("intent"):
            badges += intent_badge(out["intent"], out.get("intent_confidence") or 0) + "  "
        badges += taste_badge(out.get("use_taste", False), out.get("taste_key"))
        if out["latency_ms"]:
            badges += f'&nbsp;&nbsp;<span style="color:#666;font-size:0.8rem">⏱ {out["latency_ms"]:.0f} ms</span>'
        st.markdown(badges, unsafe_allow_html=True)

        if out.get("reference_detected"):
            st.info(f"Reference resolved: \"{out['resolved_query']}\"")

        st.markdown(
            f'<div class="answer-box">{out["answer"]}</div>',
            unsafe_allow_html=True,
        )

        if out["results"]:
            st.markdown("**Top matches**")
            render_results(out["results"])


# ── TAB 2: Debug ────────────────────────────────────────────────────────────────

with tab_debug:
    st.markdown("Full routing trace and per-result score breakdown.")

    sel_system_d = st.radio(
        "system_debug",
        SYSTEM_LABELS,
        index=4,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.caption(f"_{SYSTEM_META[sel_system_d]['desc']}_")

    query_d = st.text_input(
        "Query",
        placeholder="Jedi I rated 9+ from the prequel era",
        key="q_debug",
    )

    if query_d:
        # Debug tab uses current S5 session history so you can test follow-ups
        debug_history = (
            st.session_state.history
            if sel_system_d == "S5 — Full agent: router + mood centroids"
            else []
        )
        with st.spinner(f"Running {SYSTEM_META[sel_system_d]['short']}…"):
            out = run_system(query_d, sel_system_d, history=debug_history)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("System", out["system_short"])
        m2.metric(
            "Intent",
            f"{INTENT_EMOJI.get(out.get('intent',''), '—')} {out.get('intent','')}"
            if out.get("intent") else "— (no router)",
        )
        m3.metric("Taste", f"ON · {out['taste_key']}" if out.get("use_taste") else "OFF")
        m4.metric("Latency", f"{out['latency_ms']:.0f} ms")

        # Reference resolution display
        if out.get("reference_detected"):
            st.info(
                f"**resolve_references fired** — rewritten query: "
                f"\"{out['resolved_query']}\""
            )
        elif sel_system_d == "S5 — Full agent: router + mood centroids":
            st.caption("_resolve_references: no referential phrase detected_")

        st.markdown("**Answer**")
        st.markdown(
            f'<div class="answer-box">{out["answer"]}</div>',
            unsafe_allow_html=True,
        )

        if out["results"]:
            st.markdown("**Retrieved results (with score components)**")
            render_results(out["results"], debug=True)


# ── TAB 3: Catalogue ────────────────────────────────────────────────────────────

with tab_catalogue:
    st.markdown("Browse and filter your 70-entity Star Wars knowledge base.")

    df = load_catalogue()

    f1, f2, f3, f4 = st.columns(4)
    sel_type = f1.selectbox("Type", ["All"] + sorted(df["type"].unique().tolist()))
    sel_era  = f2.selectbox("Era",  ["All"] + sorted(df["era"].unique().tolist()))
    all_moods = sorted({
        m.strip() for moods in df["mood"].dropna() for m in moods.split(",")
    })
    sel_mood = f3.selectbox("Mood", ["All"] + all_moods)
    min_r, max_r = f4.slider("Rating", 1, 10, (1, 10))

    mask = (df["your_rating"] >= min_r) & (df["your_rating"] <= max_r)
    if sel_type != "All":
        mask &= df["type"] == sel_type
    if sel_era != "All":
        mask &= df["era"] == sel_era
    if sel_mood != "All":
        mask &= df["mood"].str.contains(sel_mood, case=False, na=False)

    filtered = df[mask].sort_values("your_rating", ascending=False).reset_index(drop=True)
    st.caption(f"{len(filtered)} of {len(df)} entities")

    CARDS_PER_ROW = 5
    for row_start in range(0, len(filtered), CARDS_PER_ROW):
        cols = st.columns(CARDS_PER_ROW)
        for col, (_, row) in zip(cols, filtered.iloc[row_start:row_start + CARDS_PER_ROW].iterrows()):
            with col:
                img = _entity_image(row["entity_id"])
                if img:
                    st.image(img, use_container_width=True)
                st.markdown(f"**{row['name']}**")
                st.caption(f"⭐ {row['your_rating']}/10 · {row['type']}")
                st.caption(f"{row['era']} · *{row['mood']}*")
                with st.expander("Your take"):
                    st.write(row["your_take"])
