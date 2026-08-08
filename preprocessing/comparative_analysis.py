"""
Comparative analysis of preprocessing / feature-extraction strategies
(assignment requirement C).

Grid: normalization (none / stem / lemmatize) x feature type (BoW / TF-IDF)
x n-gram range (unigram / unigram+bigram), scored by vocabulary size and
cross-validated classification macro-F1 against the seed_source labels.
"""

import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB

from preprocessing.feature_engineering import build_bow, build_tfidf, load_corpus

PROCESSED_DIR = "data/processed"
FIGURES_DIR = "data/processed/figures"

NORMALIZE_OPTIONS = ["none", "stem", "lemmatize"]
NGRAM_OPTIONS = {"unigram": (1, 1), "unigram+bigram": (1, 2)}
FEATURE_BUILDERS = {"bow": build_bow, "tfidf": build_tfidf}


def run_comparison(corpus: pd.DataFrame, n_splits: int = 4) -> pd.DataFrame:
    texts = corpus["text"].tolist()
    labels = corpus["seed_source"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    rows = []
    for normalize, ngram_name, feature_name in itertools.product(
        NORMALIZE_OPTIONS, NGRAM_OPTIONS, FEATURE_BUILDERS
    ):
        builder = FEATURE_BUILDERS[feature_name]
        ngram_range = NGRAM_OPTIONS[ngram_name]
        _, matrix = builder(texts, normalize=normalize, ngram_range=ngram_range, max_features=None)

        f1_scores = cross_val_score(MultinomialNB(), matrix, labels, cv=skf, scoring="f1_macro")

        rows.append({
            "normalize": normalize,
            "ngram_range": ngram_name,
            "feature_type": feature_name,
            "vocab_size": matrix.shape[1],
            "sparsity_pct": 100.0 * (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
            "mean_cv_macro_f1": float(f1_scores.mean()),
            "std_cv_macro_f1": float(f1_scores.std()),
        })

    return pd.DataFrame(rows).sort_values("mean_cv_macro_f1", ascending=False).reset_index(drop=True)


def plot_comparison(results: pd.DataFrame, out_path: str):
    results = results.copy()
    results["config"] = (
        results["normalize"] + " / " + results["ngram_range"] + " / " + results["feature_type"]
    )
    results = results.sort_values("mean_cv_macro_f1")

    plt.figure(figsize=(9, 8))
    plt.barh(results["config"], results["mean_cv_macro_f1"], xerr=results["std_cv_macro_f1"])
    plt.xlabel("Mean cross-validated macro-F1")
    plt.title("Preprocessing Strategy Comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_vocab_sizes(results: pd.DataFrame, out_path: str):
    pivot = results.pivot_table(index="normalize", columns="ngram_range", values="vocab_size", aggfunc="mean")
    pivot.plot(kind="bar", figsize=(8, 5))
    plt.ylabel("Vocabulary size")
    plt.title("Vocabulary Size by Normalization Strategy")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)

    corpus = load_corpus()
    results = run_comparison(corpus)
    results.to_csv(os.path.join(PROCESSED_DIR, "preprocessing_comparison.csv"), index=False)
    print(results.to_string(index=False))

    plot_comparison(results, os.path.join(FIGURES_DIR, "preprocessing_comparison.png"))
    plot_vocab_sizes(results, os.path.join(FIGURES_DIR, "vocab_size_comparison.png"))
    print(f"\nSaved comparison table and plots to {PROCESSED_DIR}")
