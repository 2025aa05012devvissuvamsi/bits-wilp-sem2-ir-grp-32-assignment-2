"""Evaluation dashboard (assignment requirement F)."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import plotly.express as px
import streamlit as st

from utils.streamlit_helpers import get_search_engine, index_data_exists

st.set_page_config(page_title="Evaluation", layout="wide")
st.title("Evaluation Dashboard")

if not index_data_exists():
    st.warning("Index not built yet. Go to the **Index Management** page first.")
    st.stop()

st.markdown(
    "Relevance judgments use a topic-match proxy: a document is considered relevant to a "
    "test query if its `seed_source` matches the query's target topic(s). This is a "
    "reproducible methodology suited to a small crawled corpus, not manually-curated "
    "expert judgments — see the report for the full methodology discussion."
)

from evaluation.evaluator import compare_methods
from evaluation.relevance_judgments import TEST_QUERIES, build_relevance_sets

with st.expander("Test query set"):
    for q in TEST_QUERIES:
        st.markdown(f"- **{q['query']}** — relevant topics: {', '.join(q['relevant_topics'])}")

engine = get_search_engine()


@st.cache_data(show_spinner="Running evaluation across ranking methods...")
def _run_comparison():
    judgments = build_relevance_sets()
    summary, details = compare_methods(engine, judgments)
    return summary, {k: v for k, v in details.items()}


summary, details = _run_comparison()

st.subheader("Ranking method comparison")
st.dataframe(summary, hide_index=True, use_container_width=True)

metrics_to_plot = ["MAP", "MRR", "mean_ndcg_at_5", "mean_f1"]
melted = summary.melt(id_vars="method", value_vars=metrics_to_plot, var_name="metric", value_name="score")
fig = px.bar(melted, x="method", y="score", color="metric", barmode="group")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Per-query breakdown")
method_choice = st.selectbox("Method", list(details.keys()), index=list(details.keys()).index("hybrid_pagerank"))
st.dataframe(details[method_choice], hide_index=True, use_container_width=True)
