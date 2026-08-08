"""Index management (supports assignment requirement D: efficient search over indexed collections)."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.streamlit_helpers import clear_all_caches, crawl_data_exists, get_doc_index, get_index_stats, index_data_exists

st.set_page_config(page_title="Index Management", layout="wide")
st.title("Index Management")

if not crawl_data_exists():
    st.warning("No crawled data yet. Go to the **Crawler** page first.")
    st.stop()

if st.button("Rebuild index now", type="primary"):
    with st.spinner("Rebuilding inverted index, vector index, and link graph..."):
        from indexing.index_manager import build_all
        stats = build_all()
    clear_all_caches()
    st.success(f"Index rebuilt: {stats['vocabulary_size']} terms, {stats['num_documents']} documents.")

if not index_data_exists():
    st.info("Index not built yet. Click **Rebuild index now** above.")
    st.stop()

stats = get_index_stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Documents indexed", stats["num_documents"])
c2.metric("Vocabulary size", stats["vocabulary_size"])
c3.metric("Link graph nodes", stats["graph_nodes"])
c4.metric("Link graph edges", stats["graph_edges"])
st.caption(f"TF-IDF matrix shape: {tuple(stats['tfidf_matrix_shape'])}")

st.divider()
st.subheader("Term lookup (inverted index)")

term_query = st.text_input("Look up a term (cleaned the same way as document text)")
if term_query:
    from indexing.inverted_index import load_index
    from preprocessing.text_cleaner import clean

    index = load_index()
    doc_index = get_doc_index().set_index("doc_id")

    for term in clean(term_query):
        postings = index.get(term, {})
        st.markdown(f"**'{term}'** — appears in {len(postings)} document(s)")
        if postings:
            rows = []
            for doc_id, tf in sorted(postings.items(), key=lambda x: -x[1]):
                title = doc_index.loc[doc_id, "title"] if doc_id in doc_index.index else doc_id
                rows.append({"doc_id": doc_id, "title": title, "term_frequency": tf})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.divider()
st.subheader("Most frequent terms (by document frequency)")

from indexing.inverted_index import document_frequencies, load_index

index = load_index()
df = document_frequencies(index)
top_terms = pd.DataFrame(sorted(df.items(), key=lambda x: -x[1])[:20], columns=["term", "document_frequency"])
fig = px.bar(top_terms, x="term", y="document_frequency")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Browse indexed documents")
doc_index = get_doc_index()
topic_filter = st.multiselect("Filter by topic", sorted(doc_index["seed_source"].unique()))
filtered = doc_index[doc_index["seed_source"].isin(topic_filter)] if topic_filter else doc_index
st.dataframe(filtered, hide_index=True, use_container_width=True)
