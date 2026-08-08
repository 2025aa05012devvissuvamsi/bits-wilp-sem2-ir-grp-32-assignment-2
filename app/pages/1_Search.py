"""Search interface (assignment requirement D)."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from utils.streamlit_helpers import get_search_engine, index_data_exists

st.set_page_config(page_title="Search", layout="wide")
st.title("Search Interface")

if not index_data_exists():
    st.warning("Index not built yet. Go to the **Index Management** page first.")
    st.stop()

engine = get_search_engine()

with st.sidebar:
    st.header("Query options")
    rank_method = st.selectbox("Ranking method", ["hybrid", "content", "link"], index=0)
    link_algorithm = st.selectbox("Link algorithm", ["pagerank", "hits"], index=0)
    alpha = st.slider(
        "Alpha (content weight)", 0.0, 1.0, 0.7, 0.05,
        help="Only used for hybrid ranking: final = alpha*content + (1-alpha)*link",
        disabled=rank_method != "hybrid",
    )
    boolean_mode = st.radio("Candidate filter", ["OR", "AND"], horizontal=True,
                             help="OR: doc must contain at least one query term. AND: doc must contain all.")
    expand = st.checkbox("Expand query with WordNet synonyms", value=False)
    top_k = st.slider("Number of results", 1, 20, 10)

query = st.text_input("Enter your query", placeholder="e.g. neural networks for computer vision")

if query:
    results = engine.search(
        query, top_k=top_k, expand=expand, boolean_mode=boolean_mode,
        rank_method=rank_method, link_algorithm=link_algorithm, alpha=alpha,
    )

    if results.empty:
        st.info("No matching documents found. Try a different query or switch the candidate filter to OR.")
    else:
        st.caption(f"{len(results)} result(s)")
        for _, row in results.iterrows():
            with st.container(border=True):
                st.markdown(f"**[{row['title']}]({row['url']})**  \n*Topic: {row['seed_source']}*")
                st.write(row["snippet"])
                c1, c2, c3 = st.columns(3)
                c1.metric("Content score", f"{row['content_score']:.3f}")
                c2.metric("Link score", f"{row['link_score']:.3f}")
                c3.metric("Final score", f"{row['final_score']:.3f}")

    with st.expander("Compare: content-only vs. hybrid ranking"):
        content_only, hybrid = engine.compare_rankings(query, top_k=top_k, alpha=alpha, link_algorithm=link_algorithm)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Content-only order**")
            st.dataframe(content_only[["title", "content_score"]], hide_index=True, use_container_width=True)
        with c2:
            st.markdown("**Hybrid order**")
            st.dataframe(hybrid[["title", "final_score"]], hide_index=True, use_container_width=True)
        st.caption(
            "This shows concretely how link-based authority (PageRank/HITS) can reorder results "
            "relative to pure content similarity — sometimes surfacing a highly-linked but less "
            "topically relevant document above a more relevant one."
        )
else:
    st.info("Enter a query above to search the indexed collection.")
