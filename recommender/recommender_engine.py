"""
Orchestration layer over content / graph / hybrid recommenders. This is
what the Streamlit "Recommendation panel" will call.
"""

import pandas as pd

from recommender.content_recommender import recommend_similar
from recommender.graph_recommender import recommend_via_graph
from recommender.hybrid_recommender import recommend_hybrid

METHODS = {"content", "graph", "hybrid"}


def get_recommendations(doc_id: str, method: str = "hybrid", top_k: int = 10, alpha: float = 0.6) -> pd.DataFrame:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Must be one of {METHODS}")

    if method == "content":
        return recommend_similar(doc_id, top_k=top_k)
    if method == "graph":
        return recommend_via_graph(doc_id, top_k=top_k)
    return recommend_hybrid(doc_id, top_k=top_k, alpha=alpha)


if __name__ == "__main__":
    from indexing.vector_index import load_vector_index

    _, _, doc_index = load_vector_index()
    sample_doc = doc_index.iloc[10]
    print(f"Top-5 recommendations for: {sample_doc['title']} ({sample_doc['doc_id']})\n")

    for method in ("content", "graph", "hybrid"):
        print(f"--- {method} ---")
        print(get_recommendations(sample_doc["doc_id"], method=method, top_k=5).to_string(index=False))
        print()
