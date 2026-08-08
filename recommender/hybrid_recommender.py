"""
Hybrid recommendation: alpha-weighted blend of content similarity and
graph proximity (assignment requirement E).
"""

import pandas as pd

from recommender.content_recommender import recommend_similar
from recommender.graph_recommender import recommend_via_graph
from search.ranking import normalize_scores

RESULT_COLUMNS = ["doc_id", "title", "seed_source", "content_score", "graph_score", "score"]


def recommend_hybrid(doc_id: str, top_k: int = 10, alpha: float = 0.6, index_dir: str = "data/index") -> pd.DataFrame:
    """alpha weights content similarity; (1 - alpha) weights graph proximity."""
    content_df = recommend_similar(doc_id, top_k=10_000, index_dir=index_dir)
    graph_df = recommend_via_graph(doc_id, top_k=10_000, index_dir=index_dir)

    content_scores = normalize_scores(dict(zip(content_df["doc_id"], content_df["score"])))
    graph_scores = normalize_scores(dict(zip(graph_df["doc_id"], graph_df["score"])))
    meta = content_df.set_index("doc_id")[["title", "seed_source"]]
    for doc_id_, row in graph_df.set_index("doc_id")[["title", "seed_source"]].iterrows():
        if doc_id_ not in meta.index:
            meta.loc[doc_id_] = row

    all_ids = set(content_scores) | set(graph_scores)
    rows = []
    for other_id in all_ids:
        c = content_scores.get(other_id, 0.0)
        g = graph_scores.get(other_id, 0.0)
        rows.append({
            "doc_id": other_id,
            "title": meta.loc[other_id, "title"],
            "seed_source": meta.loc[other_id, "seed_source"],
            "content_score": c,
            "graph_score": g,
            "score": alpha * c + (1 - alpha) * g,
        })

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return results.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)


if __name__ == "__main__":
    from indexing.vector_index import load_vector_index

    _, _, doc_index = load_vector_index()
    sample_doc = doc_index.iloc[0]
    print(f"Hybrid recommendations for: {sample_doc['title']} ({sample_doc['doc_id']})\n")
    print(recommend_hybrid(sample_doc["doc_id"], top_k=5).to_string(index=False))
