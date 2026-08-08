"""
Document profiling + corpus-level statistical analysis and visualizations
(assignment requirement C).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing.feature_engineering import load_corpus
from preprocessing.keyword_extraction import load_tfidf
from preprocessing.text_cleaner import clean, tokenize

PROCESSED_DIR = "data/processed"
FIGURES_DIR = "data/processed/figures"


def profile_documents(corpus: pd.DataFrame) -> pd.DataFrame:
    """Per-document statistics: length, vocabulary richness, etc."""
    rows = []
    for _, doc in corpus.iterrows():
        raw_tokens = tokenize(doc["text"])
        clean_tokens = clean(doc["text"])
        unique_clean = set(clean_tokens)
        rows.append({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "seed_source": doc["seed_source"],
            "char_count": len(doc["text"]),
            "raw_token_count": len(raw_tokens),
            "clean_token_count": len(clean_tokens),
            "unique_token_count": len(unique_clean),
            "vocab_richness": len(unique_clean) / len(clean_tokens) if clean_tokens else 0.0,
            "avg_word_length": float(np.mean([len(t) for t in raw_tokens])) if raw_tokens else 0.0,
        })
    return pd.DataFrame(rows)


def most_similar_docs(target_doc_id: str, matrix, doc_ids, top_n: int = 5) -> pd.DataFrame:
    """Cosine-similarity nearest neighbors for a document, using the TF-IDF matrix."""
    idx = doc_ids.index(target_doc_id)
    sims = cosine_similarity(matrix[idx], matrix).ravel()
    order = sims.argsort()[::-1]
    order = [i for i in order if i != idx][:top_n]
    return pd.DataFrame({
        "doc_id": [doc_ids[i] for i in order],
        "similarity": [float(sims[i]) for i in order],
    })


def zipf_plot(corpus: pd.DataFrame, out_path: str):
    """Word frequency rank vs. frequency on a log-log scale (Zipf's law check)."""
    from collections import Counter

    all_tokens = []
    for text in corpus["text"]:
        all_tokens.extend(tokenize(text))
    freqs = Counter(all_tokens).most_common()
    ranks = np.arange(1, len(freqs) + 1)
    counts = np.array([c for _, c in freqs])

    plt.figure(figsize=(7, 5))
    plt.loglog(ranks, counts, marker=".", linestyle="none")
    plt.xlabel("Rank (log)")
    plt.ylabel("Frequency (log)")
    plt.title("Zipf's Law: Term Frequency vs. Rank")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def heaps_plot(corpus: pd.DataFrame, out_path: str):
    """Vocabulary size growth vs. total tokens seen (Heaps' law check)."""
    seen_vocab = set()
    vocab_sizes = []
    token_counts = []
    total_tokens = 0
    for text in corpus["text"]:
        tokens = tokenize(text)
        total_tokens += len(tokens)
        seen_vocab.update(tokens)
        token_counts.append(total_tokens)
        vocab_sizes.append(len(seen_vocab))

    plt.figure(figsize=(7, 5))
    plt.plot(token_counts, vocab_sizes, marker="o")
    plt.xlabel("Total tokens processed")
    plt.ylabel("Vocabulary size")
    plt.title("Heaps' Law: Vocabulary Growth")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def length_distribution_plot(profile: pd.DataFrame, out_path: str):
    plt.figure(figsize=(7, 5))
    sns.histplot(profile["clean_token_count"], bins=15, kde=True)
    plt.xlabel("Clean token count per document")
    plt.title("Document Length Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def vocab_richness_by_topic_plot(profile: pd.DataFrame, out_path: str):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=profile, x="seed_source", y="vocab_richness")
    plt.xticks(rotation=30, ha="right")
    plt.title("Vocabulary Richness by Topic")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)

    corpus = load_corpus()
    profile = profile_documents(corpus)
    profile.to_csv(os.path.join(PROCESSED_DIR, "document_profiles.csv"), index=False)
    print(f"Saved profiles for {len(profile)} documents.")

    matrix, vocab, doc_ids = load_tfidf()
    similar_rows = []
    for doc_id in doc_ids:
        sims = most_similar_docs(doc_id, matrix, doc_ids, top_n=3)
        sims["source_doc_id"] = doc_id
        similar_rows.append(sims)
    pd.concat(similar_rows, ignore_index=True).to_csv(
        os.path.join(PROCESSED_DIR, "similar_documents.csv"), index=False
    )
    print("Saved nearest-neighbor similarity table.")

    zipf_plot(corpus, os.path.join(FIGURES_DIR, "zipf_law.png"))
    heaps_plot(corpus, os.path.join(FIGURES_DIR, "heaps_law.png"))
    length_distribution_plot(profile, os.path.join(FIGURES_DIR, "doc_length_distribution.png"))
    vocab_richness_by_topic_plot(profile, os.path.join(FIGURES_DIR, "vocab_richness_by_topic.png"))
    print(f"Saved corpus visualizations to {FIGURES_DIR}")
