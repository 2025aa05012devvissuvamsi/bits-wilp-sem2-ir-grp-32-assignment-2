"""
Content-based recommendation: TF-IDF cosine similarity between documents
(assignment requirement E). Reuses the same vector index built for search,
so "documents similar to this one" is consistent across the app.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from indexing.vector_index import load_vector_index

RESULT_COLUMNS = ["doc_id", "title", "seed_source", "score"]


def recommend_similar(doc_id: str, top_k: int = 10, index_dir: str = "data/index") -> pd.DataFrame:
    vectorizer, matrix, doc_index = load_vector_index(out_dir=index_dir)
    doc_id_to_pos = {d: i for i, d in enumerate(doc_index["doc_id"])}

    if doc_id not in doc_id_to_pos:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    idx = doc_id_to_pos[doc_id]
    sims = cosine_similarity(matrix[idx], matrix).ravel()

    rows = []
    for other_id, pos in doc_id_to_pos.items():
        if other_id == doc_id:
            continue
        row = doc_index.iloc[pos]
        rows.append({
            "doc_id": other_id,
            "title": row["title"],
            "seed_source": row["seed_source"],
            "score": float(sims[pos]),
        })

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return results.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)


if __name__ == "__main__":
    from indexing.vector_index import load_vector_index

    _, _, doc_index = load_vector_index()
    sample_doc = doc_index.iloc[0]
    print(f"Recommendations similar to: {sample_doc['title']} ({sample_doc['doc_id']})\n")
    print(recommend_similar(sample_doc["doc_id"], top_k=5).to_string(index=False))
