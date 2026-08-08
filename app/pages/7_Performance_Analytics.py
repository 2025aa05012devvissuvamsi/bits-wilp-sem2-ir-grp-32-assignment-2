"""
Performance analytics: (1) the preprocessing/classification comparative
analysis from Step 3, surfaced live rather than as static notebook output,
and (2) actual pipeline runtime benchmarking.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.streamlit_helpers import clear_all_caches, crawl_data_exists

st.set_page_config(page_title="Performance Analytics", layout="wide")
st.title("Performance Analytics")

if not crawl_data_exists():
    st.warning("No crawled data yet. Go to the **Crawler** page first.")
    st.stop()

PROCESSED_DIR = "data/processed"
FIGURES_DIR = "data/processed/figures"

tab1, tab2, tab3, tab4 = st.tabs([
    "Preprocessing Comparison", "Classification", "Corpus Statistics", "Runtime Benchmark",
])

with tab1:
    st.subheader("Preprocessing strategy comparison")
    comparison_csv = os.path.join(PROCESSED_DIR, "preprocessing_comparison.csv")
    if os.path.exists(comparison_csv):
        df = pd.read_csv(comparison_csv)
        st.dataframe(df, hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.image(os.path.join(FIGURES_DIR, "preprocessing_comparison.png"), use_container_width=True)
        with c2:
            st.image(os.path.join(FIGURES_DIR, "vocab_size_comparison.png"), use_container_width=True)
    else:
        st.info("Run `python -m preprocessing.comparative_analysis` to generate this.")

with tab2:
    st.subheader("Document classification results")
    cv_csv = os.path.join(PROCESSED_DIR, "classification_cv_results.csv")
    if os.path.exists(cv_csv):
        st.dataframe(pd.read_csv(cv_csv), hide_index=True, use_container_width=True)
        st.image(os.path.join(FIGURES_DIR, "confusion_matrix.png"), use_container_width=False)
        report_path = os.path.join(PROCESSED_DIR, "classification_report.txt")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                st.code(f.read())
    else:
        st.info("Run `python -m preprocessing.classifier` to generate this.")

with tab3:
    st.subheader("Corpus statistics")
    for fname, caption in [
        ("zipf_law.png", "Zipf's Law: term frequency vs. rank"),
        ("heaps_law.png", "Heaps' Law: vocabulary growth"),
        ("doc_length_distribution.png", "Document length distribution"),
        ("vocab_richness_by_topic.png", "Vocabulary richness by topic"),
    ]:
        path = os.path.join(FIGURES_DIR, fname)
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)

    st.subheader("Word clouds by topic")
    wordcloud_paths = sorted(glob.glob(os.path.join(FIGURES_DIR, "wordcloud_*.png")))
    if wordcloud_paths:
        cols = st.columns(2)
        for i, path in enumerate(wordcloud_paths):
            topic = os.path.basename(path).replace("wordcloud_", "").replace(".png", "").replace("_", " ")
            with cols[i % 2]:
                st.image(path, caption=topic, use_container_width=True)
    else:
        st.info("Run `python -m preprocessing.keyword_extraction` to generate word clouds.")

with tab4:
    st.subheader("Pipeline runtime benchmark")
    st.caption(
        "Times each pipeline stage on the current corpus. Rebuilds the index from the "
        "existing crawled data (safe — doesn't re-crawl or change the document set)."
    )
    if st.button("Run benchmark", type="primary"):
        from indexing.index_manager import build_all
        from preprocessing.feature_engineering import build_tfidf, load_corpus
        from search.search_engine import SearchEngine
        from evaluation.relevance_judgments import TEST_QUERIES

        timings = {}

        t0 = time.time()
        corpus = load_corpus()
        timings["Load corpus"] = time.time() - t0

        t0 = time.time()
        build_tfidf(corpus["text"].tolist())
        timings["Build TF-IDF (preprocessing)"] = time.time() - t0

        t0 = time.time()
        build_all()
        timings["Rebuild full index"] = time.time() - t0

        t0 = time.time()
        engine = SearchEngine()
        timings["Load search engine"] = time.time() - t0

        t0 = time.time()
        for q in TEST_QUERIES:
            engine.search(q["query"], top_k=10)
        timings["Avg. search query"] = (time.time() - t0) / len(TEST_QUERIES)

        t0 = time.time()
        sample_doc = engine.doc_index.iloc[0]["doc_id"]
        from recommender.recommender_engine import get_recommendations
        get_recommendations(sample_doc, method="hybrid", top_k=10)
        timings["Recommendation generation"] = time.time() - t0

        clear_all_caches()

        timing_df = pd.DataFrame(
            [{"stage": k, "seconds": v} for k, v in timings.items()]
        ).sort_values("seconds", ascending=False)
        st.dataframe(timing_df, hide_index=True, use_container_width=True)
        fig = px.bar(timing_df, x="stage", y="seconds")
        st.plotly_chart(fig, use_container_width=True)
