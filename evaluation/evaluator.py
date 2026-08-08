"""
Runs the test query set through the search engine under different ranking
configurations and reports precision/recall/F1/P@K/R@K/MAP/MRR/NDCG,
including a comparative table across ranking methods (assignment
requirement F: "provide comparative analysis using tables and
visualizations").
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from evaluation.metrics import (
    average_precision, f1_score, ndcg_at_k, precision, precision_at_k,
    recall, recall_at_k, reciprocal_rank,
)
from evaluation.relevance_judgments import build_relevance_sets
from search.search_engine import SearchEngine

EVAL_DIR = "data/evaluation"
FIGURES_DIR = "data/evaluation/figures"
TOP_K = 10

METHOD_CONFIGS = {
    "content_only": {"rank_method": "content"},
    "pagerank_only": {"rank_method": "link", "link_algorithm": "pagerank"},
    "hits_only": {"rank_method": "link", "link_algorithm": "hits"},
    "hybrid_pagerank": {"rank_method": "hybrid", "link_algorithm": "pagerank", "alpha": 0.7},
    "hybrid_hits": {"rank_method": "hybrid", "link_algorithm": "hits", "alpha": 0.7},
}


def evaluate_method(engine: SearchEngine, judgments: list, method_kwargs: dict, top_k: int = TOP_K) -> pd.DataFrame:
    rows = []
    for j in judgments:
        results = engine.search(j["query"], top_k=top_k, **method_kwargs)
        ranked_ids = results["doc_id"].tolist()
        relevant_ids = j["relevant_doc_ids"]

        p = precision(ranked_ids, relevant_ids)
        r = recall(ranked_ids, relevant_ids)
        rows.append({
            "query": j["query"],
            "num_relevant": len(relevant_ids),
            "num_retrieved": len(ranked_ids),
            "precision": p,
            "recall": r,
            "f1": f1_score(p, r),
            "precision_at_5": precision_at_k(ranked_ids, relevant_ids, 5),
            "recall_at_5": recall_at_k(ranked_ids, relevant_ids, 5),
            "average_precision": average_precision(ranked_ids, relevant_ids),
            "reciprocal_rank": reciprocal_rank(ranked_ids, relevant_ids),
            "ndcg_at_5": ndcg_at_k(ranked_ids, relevant_ids, 5),
            "ndcg_at_10": ndcg_at_k(ranked_ids, relevant_ids, 10),
        })
    return pd.DataFrame(rows)


def compare_methods(engine: SearchEngine, judgments: list, top_k: int = TOP_K) -> pd.DataFrame:
    summary_rows = []
    per_method_details = {}
    for method_name, kwargs in METHOD_CONFIGS.items():
        detail = evaluate_method(engine, judgments, kwargs, top_k=top_k)
        per_method_details[method_name] = detail
        summary_rows.append({
            "method": method_name,
            "MAP": detail["average_precision"].mean(),
            "MRR": detail["reciprocal_rank"].mean(),
            "mean_precision_at_5": detail["precision_at_5"].mean(),
            "mean_recall_at_5": detail["recall_at_5"].mean(),
            "mean_f1": detail["f1"].mean(),
            "mean_ndcg_at_5": detail["ndcg_at_5"].mean(),
            "mean_ndcg_at_10": detail["ndcg_at_10"].mean(),
        })
    return pd.DataFrame(summary_rows), per_method_details


def plot_comparison(summary: pd.DataFrame, out_path: str):
    metrics_to_plot = ["MAP", "MRR", "mean_ndcg_at_5", "mean_f1"]
    plot_df = summary.set_index("method")[metrics_to_plot]

    plot_df.plot(kind="bar", figsize=(10, 6))
    plt.ylabel("Score")
    plt.title("Ranking Method Comparison Across IR Metrics")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


if __name__ == "__main__":
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    engine = SearchEngine()
    judgments = build_relevance_sets()

    summary, details = compare_methods(engine, judgments)
    summary.to_csv(os.path.join(EVAL_DIR, "method_comparison_summary.csv"), index=False)
    print(summary.to_string(index=False))

    details["hybrid_pagerank"].to_csv(os.path.join(EVAL_DIR, "per_query_hybrid_pagerank.csv"), index=False)

    plot_comparison(summary, os.path.join(FIGURES_DIR, "method_comparison.png"))
    print(f"\nSaved evaluation results to {EVAL_DIR}")
