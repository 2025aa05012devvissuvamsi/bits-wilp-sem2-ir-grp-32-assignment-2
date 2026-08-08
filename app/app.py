"""
Dashboard (Home page) for the IR Assignment 2 Streamlit application.

Run from the project root with: streamlit run app/app.py
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.streamlit_helpers import crawl_data_exists, get_index_stats, get_metadata, index_data_exists

st.set_page_config(page_title="IR Assignment 2 - Dashboard", layout="wide")

st.title("Information Retrieval System - Dashboard")
st.caption(
    "End-to-end IR pipeline: crawling -> preprocessing/mining -> indexing -> "
    "search & ranking -> recommendation -> evaluation."
)

if not crawl_data_exists():
    st.warning(
        "No crawled data found yet. Go to the **Crawler** page in the sidebar to run the "
        "initial crawl before using the other pages."
    )
    st.stop()

meta = get_metadata()
stored = meta[~meta["is_duplicate"]]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Pages processed", len(meta))
col2.metric("Documents stored", len(stored))
col3.metric("Duplicates detected", int(meta["is_duplicate"].sum()))
col4.metric("Seed topics", stored["seed_source"].nunique())

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Documents per topic")
    topic_counts = stored["seed_source"].value_counts().reset_index()
    topic_counts.columns = ["seed_source", "count"]
    fig = px.bar(topic_counts, x="seed_source", y="count", labels={"seed_source": "Topic", "count": "Documents"})
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Duplicate breakdown")
    dup_counts = meta["duplicate_type"].value_counts().reset_index()
    dup_counts.columns = ["duplicate_type", "count"]
    fig = px.pie(dup_counts, names="duplicate_type", values="count")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Index status")
if index_data_exists():
    stats = get_index_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vocabulary size", stats["vocabulary_size"])
    c2.metric("Avg postings/term", round(stats["avg_postings_per_term"], 2))
    c3.metric("Link graph nodes", stats["graph_nodes"])
    c4.metric("Link graph edges", stats["graph_edges"])
else:
    st.info("Index not built yet. Go to the **Index Management** page to build it.")

crawl_config_path = "data/metadata/crawl_config.json"
if os.path.exists(crawl_config_path):
    with open(crawl_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with st.expander("Last crawl configuration"):
        st.json(config)

st.divider()
st.subheader("Pipeline stages")
st.markdown(
    """
    Use the sidebar to navigate:
    - **Search** — run ranked queries against the indexed collection
    - **Crawler** — configure seeds/depth and (re)crawl the source data
    - **Index Management** — inspect and rebuild the inverted index, vector index, and link graph
    - **Ranking** — visualize PageRank/HITS and see how link-based ranking changes results
    - **Recommendations** — content-based, graph-based, and hybrid document recommendations
    - **Evaluation** — precision/recall/MAP/MRR/NDCG across ranking methods
    - **Performance Analytics** — preprocessing/classification comparisons and pipeline timing
    """
)
