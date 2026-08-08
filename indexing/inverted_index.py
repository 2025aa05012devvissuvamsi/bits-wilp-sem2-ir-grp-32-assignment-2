"""
Classic inverted index: term -> postings list (assignment requirement D
support — efficient candidate retrieval over the indexed collection).

Complements the dense TF-IDF matrix in vector_index.py: the inverted index
answers "which docs contain these terms" cheaply (boolean AND/OR), while
the vector index ranks those candidates by similarity.
"""

import json
import os
from collections import Counter, defaultdict

from preprocessing.feature_engineering import load_corpus
from preprocessing.text_cleaner import clean

INDEX_DIR = "data/index"


def build_inverted_index(corpus) -> dict:
    """
    Returns {term: {doc_id: term_frequency}}.
    corpus must have 'doc_id' and 'text' columns.
    """
    index = defaultdict(dict)
    for _, doc in corpus.iterrows():
        tokens = clean(doc["text"])
        tf = Counter(tokens)
        for term, freq in tf.items():
            index[term][doc["doc_id"]] = freq
    return dict(index)


def document_frequencies(index: dict) -> dict:
    return {term: len(postings) for term, postings in index.items()}


def save_index(index: dict, out_dir: str = INDEX_DIR, name: str = "inverted_index"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)


def load_index(out_dir: str = INDEX_DIR, name: str = "inverted_index") -> dict:
    with open(os.path.join(out_dir, f"{name}.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def get_postings(index: dict, term: str) -> dict:
    """{doc_id: term_frequency} for a single (already-cleaned) term."""
    return index.get(term, {})


def docs_matching_all(index: dict, terms: list) -> set:
    """Boolean AND: doc_ids containing every term."""
    if not terms:
        return set()
    result = set(get_postings(index, terms[0]).keys())
    for term in terms[1:]:
        result &= set(get_postings(index, term).keys())
    return result


def docs_matching_any(index: dict, terms: list) -> set:
    """Boolean OR: doc_ids containing at least one term."""
    result = set()
    for term in terms:
        result |= set(get_postings(index, term).keys())
    return result


if __name__ == "__main__":
    corpus = load_corpus()
    index = build_inverted_index(corpus)
    save_index(index)

    df = document_frequencies(index)
    print(f"Vocabulary size: {len(index)}")
    print(f"Indexed documents: {len(corpus)}")
    print(f"Top 10 most common terms: {sorted(df.items(), key=lambda x: -x[1])[:10]}")
