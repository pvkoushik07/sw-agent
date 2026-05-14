"""Streamlit UI. Run: streamlit run src/app.py"""
from __future__ import annotations

import streamlit as st

from src import config
from src.agent import run_agent

st.set_page_config(page_title="Taste-Aware Star Wars Agent", layout="wide")
st.title("⭐ Taste-Aware Star Wars Universe Agent")
st.caption("UQ INFS4205/7205 — A3")

tab_chat, tab_debug = st.tabs(["Chat", "Debug"])


def _render_results(results, debug: bool = False):
    if not results:
        st.warning("No results.")
        return
    cols = st.columns(min(len(results), 5))
    for col, r in zip(cols, results):
        with col:
            meta = r.metadata
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                img_path = config.IMAGES_DIR / f"{meta['entity_id']}{ext}"
                if img_path.exists():
                    st.image(str(img_path), use_column_width=True)
                    break
            st.markdown(f"**{meta['name']}**")
            st.caption(f"⭐ {meta['your_rating']}/10 · {meta['type']} · {meta['era']}")
            st.caption(f"_{meta['mood']}_")
            with st.expander("Your take"):
                st.write(meta["your_take"])
            if debug:
                st.code(
                    f"score:  {r.final_score:.3f}\n"
                    f"  q_sim: {r.components['query_sim']:.3f}\n"
                    f"  taste: {r.components['taste_align']:.3f}\n"
                    f"  meta:  {r.components['meta_score']:.3f}\n"
                    f"  img:   {r.components['image_sim']:.3f}",
                    language="text",
                )


with tab_chat:
    query = st.text_input(
        "Ask about your Star Wars catalogue",
        placeholder="a deeply tragic character arc",
        key="q_chat",
    )
    if query:
        with st.spinner("Thinking..."):
            out = run_agent(query)
        st.markdown(f"#### Answer\n{out['answer']}")
        st.markdown("---")
        st.markdown("#### Top matches")
        _render_results(out.get("results", []))


with tab_debug:
    query = st.text_input(
        "Query (debug mode)",
        placeholder="Jedi I rated 9+ from the prequel era",
        key="q_debug",
    )
    if query:
        with st.spinner("Thinking..."):
            out = run_agent(query)

        c1, c2, c3 = st.columns(3)
        c1.metric("Intent", out.get("intent", "?"))
        c2.metric("Taste applied", "Yes" if out.get("use_taste") else "No")
        c3.metric("Total latency",
                  f"{out.get('trace', {}).get('total_ms', 0):.0f} ms")

        st.markdown("#### Node timings")
        st.json(out.get("trace", {}))

        st.markdown("#### Answer")
        st.write(out["answer"])

        st.markdown("#### Retrieved (with score components)")
        _render_results(out.get("results", []), debug=True)
