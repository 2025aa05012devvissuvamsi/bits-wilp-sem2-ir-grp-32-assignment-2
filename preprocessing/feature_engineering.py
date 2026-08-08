"""
Feature engineering: corpus loading + Bag-of-Words / TF-IDF vectorization
(assignment requirement C).

Engineered features are cached to data/processed/ so the indexing/search
step can reuse them without recomputing.
"""

import os

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from preprocessing.text_cleaner import clean_to_string

RAW_DIR = "data/raw"
META_PATH = "data/metadata/metadata.csv"
PROCESSED_DIR = "data/processed"


def load_corpus(raw_dir: str = RAW_DIR, meta_path: str = META_PATH) -> pd.DataFrame:
    """
    Load the non-duplicate stored documents, joined with their metadata.
    Returns a DataFrame with columns: doc_id, title, seed_source, text, ...
    """
    meta = pd.read_csv(meta_path)
    stored = meta[~meta["is_duplicate"]].copy()

    texts = []
    for raw_file in stored["raw_file"]:
        with open(os.path.join(raw_dir, raw_file), "r", encoding="utf-8") as f:
            texts.append(f.read())
    stored["text"] = texts
    return stored.reset_index(drop=True)


def build_bow(texts, remove_stopwords=True, normalize="lemmatize", ngram_range=(1, 1), max_features=5000):
    """Bag-of-Words (raw counts) feature matrix."""
    cleaned = [clean_to_string(t, remove_stopwords=remove_stopwords, normalize=normalize) for t in texts]
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features)
    matrix = vectorizer.fit_transform(cleaned)
    return vectorizer, matrix


def build_tfidf(texts, remove_stopwords=True, normalize="lemmatize", ngram_range=(1, 1), max_features=5000):
    """TF-IDF weighted feature matrix."""
    cleaned = [clean_to_string(t, remove_stopwords=remove_stopwords, normalize=normalize) for t in texts]
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    matrix = vectorizer.fit_transform(cleaned)
    return vectorizer, matrix


def save_features(doc_ids, vectorizer, matrix, name: str, out_dir: str = PROCESSED_DIR):
    """Persist a feature matrix + vocabulary + doc_id alignment to data/processed/."""
    os.makedirs(out_dir, exist_ok=True)
    sparse.save_npz(os.path.join(out_dir, f"{name}_matrix.npz"), matrix)
    pd.Series(vectorizer.get_feature_names_out()).to_csv(
        os.path.join(out_dir, f"{name}_vocab.csv"), index=False, header=["term"]
    )
    pd.Series(doc_ids).to_csv(os.path.join(out_dir, f"{name}_doc_ids.csv"), index=False, header=["doc_id"])


if __name__ == "__main__":
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} documents.")

    tfidf_vectorizer, tfidf_matrix = build_tfidf(corpus["text"].tolist())
    print(f"TF-IDF matrix: {tfidf_matrix.shape}")
    save_features(corpus["doc_id"], tfidf_vectorizer, tfidf_matrix, "tfidf")

    bow_vectorizer, bow_matrix = build_bow(corpus["text"].tolist())
    print(f"BoW matrix: {bow_matrix.shape}")
    save_features(corpus["doc_id"], bow_vectorizer, bow_matrix, "bow")

    corpus[["doc_id", "title", "seed_source", "depth"]].to_csv(
        os.path.join(PROCESSED_DIR, "corpus_index.csv"), index=False
    )
    print("Saved engineered features to data/processed/")
