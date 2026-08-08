"""
Keyword extraction via TF-IDF ranking, plus word-cloud generation
(assignment requirement C).

Reuses the TF-IDF matrix already cached by feature_engineering.py in
data/processed/ instead of recomputing it.
"""

import os

import pandas as pd
from scipy import sparse
from wordcloud import WordCloud

PROCESSED_DIR = "data/processed"
FIGURES_DIR = "data/processed/figures"


def load_tfidf(processed_dir: str = PROCESSED_DIR):
    matrix = sparse.load_npz(os.path.join(processed_dir, "tfidf_matrix.npz"))
    vocab = pd.read_csv(os.path.join(processed_dir, "tfidf_vocab.csv"))["term"].tolist()
    doc_ids = pd.read_csv(os.path.join(processed_dir, "tfidf_doc_ids.csv"))["doc_id"].tolist()
    return matrix, vocab, doc_ids


def top_keywords_for_doc(matrix, vocab, row_index: int, top_n: int = 10):
    """Top-N TF-IDF terms for a single document row."""
    row = matrix.getrow(row_index).toarray().ravel()
    top_idx = row.argsort()[::-1][:top_n]
    return [(vocab[i], float(row[i])) for i in top_idx if row[i] > 0]


def top_keywords_all(matrix, vocab, doc_ids, top_n: int = 10) -> pd.DataFrame:
    """Top-N keywords per document, as a tidy DataFrame."""
    rows = []
    for i, doc_id in enumerate(doc_ids):
        for term, score in top_keywords_for_doc(matrix, vocab, i, top_n=top_n):
            rows.append({"doc_id": doc_id, "term": term, "tfidf_score": score})
    return pd.DataFrame(rows)


def top_keywords_by_group(matrix, vocab, doc_ids, groups: pd.Series, top_n: int = 15) -> pd.DataFrame:
    """
    Top-N keywords per group (e.g. per seed_source topic), ranked by mean
    TF-IDF weight across that group's documents.
    """
    dense = matrix.toarray()
    df = pd.DataFrame(dense, columns=vocab, index=doc_ids)
    df["_group"] = groups.values

    rows = []
    for group_name, group_df in df.groupby("_group"):
        means = group_df.drop(columns="_group").mean(axis=0)
        top = means.sort_values(ascending=False).head(top_n)
        for term, score in top.items():
            rows.append({"group": group_name, "term": term, "mean_tfidf": float(score)})
    return pd.DataFrame(rows)


def generate_wordcloud(freqs: dict, out_path: str, title: str = ""):
    """Render a word cloud from a {term: weight} dict and save it as PNG."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis")
    wc.generate_from_frequencies(freqs)
    wc.to_file(out_path)


if __name__ == "__main__":
    matrix, vocab, doc_ids = load_tfidf()
    corpus_index = pd.read_csv(os.path.join(PROCESSED_DIR, "corpus_index.csv"))

    per_doc = top_keywords_all(matrix, vocab, doc_ids, top_n=10)
    per_doc.to_csv(os.path.join(PROCESSED_DIR, "keywords_per_doc.csv"), index=False)
    print(f"Saved per-doc keywords: {len(per_doc)} rows")

    per_group = top_keywords_by_group(matrix, vocab, doc_ids, corpus_index["seed_source"], top_n=15)
    per_group.to_csv(os.path.join(PROCESSED_DIR, "keywords_per_topic.csv"), index=False)
    print(f"Saved per-topic keywords: {len(per_group)} rows")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for group_name, group_rows in per_group.groupby("group"):
        freqs = dict(zip(group_rows["term"], group_rows["mean_tfidf"]))
        safe_name = "".join(c if c.isalnum() else "_" for c in group_name)
        generate_wordcloud(freqs, os.path.join(FIGURES_DIR, f"wordcloud_{safe_name}.png"), title=group_name)
    print(f"Saved word clouds to {FIGURES_DIR}")
