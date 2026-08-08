"""
Orchestration layer over inverted_index / vector_index / link_graph.
This is what the Streamlit "Index management" tab will call to rebuild
the index and display index health stats.
"""

import json
import os

from preprocessing.feature_engineering import load_corpus
from indexing.inverted_index import build_inverted_index, document_frequencies, save_index
from indexing.link_graph import build_link_graph, save_graph
from indexing.vector_index import build_vector_index

INDEX_DIR = "data/index"


def build_all(index_dir: str = INDEX_DIR) -> dict:
    """Rebuild every index artifact from the current crawled corpus and return summary stats."""
    corpus = load_corpus()

    inv_index = build_inverted_index(corpus)
    save_index(inv_index, out_dir=index_dir)

    vectorizer, matrix = build_vector_index(corpus, out_dir=index_dir)

    graph = build_link_graph()
    save_graph(graph, out_dir=index_dir)

    stats = {
        "num_documents": len(corpus),
        "vocabulary_size": len(inv_index),
        "avg_postings_per_term": (
            sum(len(p) for p in inv_index.values()) / len(inv_index) if inv_index else 0
        ),
        "tfidf_matrix_shape": list(matrix.shape),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
    }

    os.makedirs(index_dir, exist_ok=True)
    with open(os.path.join(index_dir, "index_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def get_stats(index_dir: str = INDEX_DIR) -> dict:
    """Load cached stats without rebuilding (fast path for the dashboard)."""
    stats_path = os.path.join(index_dir, "index_stats.json")
    if not os.path.exists(stats_path):
        return build_all(index_dir)
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_exists(index_dir: str = INDEX_DIR) -> bool:
    required = ["inverted_index.json", "tfidf_vectorizer.joblib", "tfidf_doc_matrix.npz", "link_graph.graphml"]
    return all(os.path.exists(os.path.join(index_dir, f)) for f in required)


if __name__ == "__main__":
    stats = build_all()
    print(json.dumps(stats, indent=2))
