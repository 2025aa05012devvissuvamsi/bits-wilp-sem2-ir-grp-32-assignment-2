"""
Shared Streamlit resource/data loaders, cached so navigating between pages
doesn't re-read disk or refit the vectorizer every time. clear_all_caches()
is called after any action that mutates the corpus or index (crawl,
rebuild) so stale objects never linger in a page that hasn't rerun yet.
"""

import os

import pandas as pd
import streamlit as st

META_PATH = "data/metadata/metadata.csv"
LINKS_PATH = "data/metadata/links.csv"
CRAWL_CONFIG_PATH = "data/metadata/crawl_config.json"
INDEX_DIR = "data/index"


def crawl_data_exists() -> bool:
    return os.path.exists(META_PATH)


def index_data_exists() -> bool:
    from indexing.index_manager import index_exists
    return index_exists(INDEX_DIR)


@st.cache_data(show_spinner=False)
def get_metadata() -> pd.DataFrame:
    return pd.read_csv(META_PATH)


@st.cache_data(show_spinner=False)
def get_doc_index() -> pd.DataFrame:
    from indexing.vector_index import load_vector_index
    _, _, doc_index = load_vector_index(out_dir=INDEX_DIR)
    return doc_index


@st.cache_resource(show_spinner="Loading search engine...")
def get_search_engine():
    from search.search_engine import SearchEngine
    return SearchEngine(index_dir=INDEX_DIR)


def get_index_stats() -> dict:
    from indexing.index_manager import get_stats
    return get_stats(INDEX_DIR)


def clear_all_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
