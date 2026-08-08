"""
Graph-based recommendation: personalized PageRank over the intra-corpus
link graph, seeded at the source document (assignment requirement E).

Unlike search/ranking.py (which ranks by directed authority for query
relevance), recommendation cares about structural closeness regardless of
link direction, so the graph is treated as undirected here.
"""

import pandas as pd
import networkx as nx

from indexing.link_graph import load_graph
from indexing.vector_index import load_vector_index

RESULT_COLUMNS = ["doc_id", "title", "seed_source", "score"]


def recommend_via_graph(doc_id: str, top_k: int = 10, index_dir: str = "data/index") -> pd.DataFrame:
    graph = load_graph().to_undirected()
    _, _, doc_index = load_vector_index(out_dir=index_dir)
    title_by_doc = doc_index.set_index("doc_id")[["title", "seed_source"]]

    if doc_id not in graph:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    personalization = {n: 0.0 for n in graph.nodes}
    personalization[doc_id] = 1.0
    scores = nx.pagerank(graph, personalization=personalization)

    rows = []
    for other_id, score in scores.items():
        if other_id == doc_id or other_id not in title_by_doc.index:
            continue
        rows.append({
            "doc_id": other_id,
            "title": title_by_doc.loc[other_id, "title"],
            "seed_source": title_by_doc.loc[other_id, "seed_source"],
            "score": float(score),
        })

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return results.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)


if __name__ == "__main__":
    from indexing.vector_index import load_vector_index

    _, _, doc_index = load_vector_index()
    sample_doc = doc_index.iloc[0]
    print(f"Graph-based recommendations for: {sample_doc['title']} ({sample_doc['doc_id']})\n")
    print(recommend_via_graph(sample_doc["doc_id"], top_k=5).to_string(index=False))
