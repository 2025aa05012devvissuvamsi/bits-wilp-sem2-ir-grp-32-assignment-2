"""
Crawling interface (assignment requirement B).

Re-crawling changes the document set, which makes the index/search/
recommender/evaluation results stale. To avoid ever leaving the app in an
inconsistent state, "Run Crawl" always cascades into an index rebuild in
the same action.
"""

import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from crawler.wiki_crawler import DEFAULT_SEEDS, WikiCrawler
from utils.streamlit_helpers import clear_all_caches, crawl_data_exists, get_metadata

st.set_page_config(page_title="Crawler", layout="wide")
st.title("Crawling Interface")

st.markdown(
    "Crawls Wikipedia breadth-first from multiple seed topics, round-robin across seeds "
    "so no single seed's link count can starve the others, with duplicate URL/document "
    "handling and retry+backoff on rate limiting."
)

if crawl_data_exists():
    meta = get_metadata()
    st.info(f"Current corpus: {len(meta)} pages processed, {int((~meta['is_duplicate']).sum())} stored.")

st.warning(
    "Running a new crawl replaces the current document set. The index, search engine, "
    "recommender, and evaluation results all depend on it, so this action will "
    "**automatically rebuild the index afterward** to keep everything consistent."
)

with st.form("crawl_config"):
    seeds_text = st.text_area(
        "Seed topics (one per line)", value="\n".join(DEFAULT_SEEDS), height=160,
    )
    c1, c2, c3 = st.columns(3)
    max_depth = c1.number_input("Max crawl depth", min_value=0, max_value=3, value=1)
    max_pages = c2.number_input("Max pages", min_value=5, max_value=500, value=50, step=5)
    near_dup_threshold = c3.slider("Near-duplicate Jaccard threshold", 0.5, 1.0, 0.85, 0.01)
    submitted = st.form_submit_button("Run Crawl + Rebuild Index", type="primary")

if submitted:
    seeds = [s.strip() for s in seeds_text.splitlines() if s.strip()]
    if not seeds:
        st.error("Provide at least one seed topic.")
        st.stop()

    progress_bar = st.progress(0.0, text="Starting crawl...")
    log_placeholder = st.empty()
    log_lines = []

    def progress_callback(msg: str):
        log_lines.append(msg)
        log_placeholder.code("\n".join(log_lines[-15:]))
        match = re.match(r"\[(\d+)/(\d+)\]", msg)
        if match:
            current, total = int(match.group(1)), int(match.group(2))
            progress_bar.progress(min(current / total, 1.0), text=f"Crawling... {current}/{total}")

    crawler = WikiCrawler(seeds=seeds, max_depth=int(max_depth), max_pages=int(max_pages),
                           near_dup_threshold=near_dup_threshold)
    result = crawler.crawl(progress_callback=progress_callback)
    progress_bar.progress(1.0, text="Crawl complete.")
    st.success(
        f"Crawl summary: {result['total_processed']} processed, "
        f"{result['stored']} stored, {result['duplicates']} duplicates."
    )

    with st.spinner("Rebuilding index (inverted index, vector index, link graph)..."):
        from indexing.index_manager import build_all
        stats = build_all()
    st.success(f"Index rebuilt: {stats['vocabulary_size']} terms, {stats['graph_edges']} link-graph edges.")

    clear_all_caches()
    st.info("Caches cleared. Other pages will now reflect the new corpus.")
