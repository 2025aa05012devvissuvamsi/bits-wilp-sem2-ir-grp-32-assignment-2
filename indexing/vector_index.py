"""
Vector-space index: persists the FITTED TF-IDF vectorizer (not just its
vocabulary) so query strings can be projected into the same vector space
at search time, plus the document-term matrix it was fit on.

preprocessing/feature_engineering.py already builds this matrix for the
classifier/comparative-analysis steps, but only saves the vocabulary list
there (sufficient for offline analysis). Search needs the live vectorizer
object to transform unseen query text consistently, hence the separate
joblib artifact here.
"""

import os

import joblib
import pandas as pd
from scipy import sparse

from preprocessing.feature_engineering import build_tfidf, load_corpus
from preprocessing.text_cleaner import clean_to_string

INDEX_DIR = "data/index"


def build_vector_index(corpus, out_dir: str = INDEX_DIR):
    os.makedirs(out_dir, exist_ok=True)
    vectorizer, matrix = build_tfidf(corpus["text"].tolist(), max_features=None)

    joblib.dump(vectorizer, os.path.join(out_dir, "tfidf_vectorizer.joblib"))
    sparse.save_npz(os.path.join(out_dir, "tfidf_doc_matrix.npz"), matrix)
    corpus[["doc_id", "title", "seed_source", "depth"]].to_csv(
        os.path.join(out_dir, "doc_index.csv"), index=False
    )
    return vectorizer, matrix


def load_vector_index(out_dir: str = INDEX_DIR):
    vectorizer = joblib.load(os.path.join(out_dir, "tfidf_vectorizer.joblib"))
    matrix = sparse.load_npz(os.path.join(out_dir, "tfidf_doc_matrix.npz"))
    doc_index = pd.read_csv(os.path.join(out_dir, "doc_index.csv"))
    return vectorizer, matrix, doc_index


def vectorize_query(vectorizer, query: str):
    """Project a raw query string into the fitted TF-IDF space."""
    cleaned = clean_to_string(query)
    return vectorizer.transform([cleaned])


if __name__ == "__main__":
    corpus = load_corpus()
    vectorizer, matrix = build_vector_index(corpus)
    print(f"Vector index built: {matrix.shape[0]} docs x {matrix.shape[1]} terms")

    q_vec = vectorize_query(vectorizer, "neural networks for computer vision")
    print(f"Sample query vector nnz: {q_vec.nnz}")
