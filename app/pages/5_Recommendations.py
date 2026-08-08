"""Recommendation panel (assignment requirement E)."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import plotly.express as px
import streamlit as st

from utils.streamlit_helpers import get_doc_index, index_data_exists

st.set_page_config(page_title="Recommendations", layout="wide")
st.title("Recommendation Panel")

if not index_data_exists():
    st.warning("Index not built yet. Go to the **Index Management** page first.")
    st.stop()

doc_index = get_doc_index().sort_values("title")
title_to_id = dict(zip(doc_index["title"], doc_index["doc_id"]))

with st.sidebar:
    st.header("Options")
    method = st.selectbox("Recommendation method", ["hybrid", "content", "graph"], index=0)
    top_k = st.slider("Number of recommendations", 1, 20, 10)
    alpha = st.slider(
        "Alpha (content weight)", 0.0, 1.0, 0.6, 0.05,
        help="Only used for hybrid: score = alpha*content_similarity + (1-alpha)*graph_proximity",
        disabled=method != "hybrid",
    )

selected_title = st.selectbox("Pick a document to get recommendations for", doc_index["title"].tolist())

if selected_title:
    doc_id = title_to_id[selected_title]
    from recommender.recommender_engine import get_recommendations

    recs = get_recommendations(doc_id, method=method, top_k=top_k, alpha=alpha)

    if recs.empty:
        st.info("No recommendations found for this document (it may be isolated in the link graph).")
    else:
        st.caption(f"Top-{len(recs)} recommendations ({method}) for '{selected_title}'")
        st.dataframe(recs, hide_index=True, use_container_width=True)

        score_col = "score"
        fig = px.bar(recs, x="title", y=score_col, color="seed_source")
        st.plotly_chart(fig, use_container_width=True)
