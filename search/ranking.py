"""
Link-based authority ranking: PageRank and HITS over the intra-corpus link
graph built in indexing/link_graph.py (assignment requirement D: "use any
of the algorithm (PageRank/HITS) and display how ranking is important").
"""

import networkx as nx


def compute_pagerank(graph: nx.DiGraph) -> dict:
    return nx.pagerank(graph)


def compute_hits(graph: nx.DiGraph, max_iter: int = 1000) -> tuple:
    """Returns (hub_scores, authority_scores). Authority scores are the
    relevant signal for ranking search results (pages linked-to by hubs)."""
    try:
        hubs, authorities = nx.hits(graph, max_iter=max_iter, normalized=True)
    except nx.PowerIterationFailedConvergence:
        hubs, authorities = nx.hits(graph, max_iter=max_iter, normalized=True, tol=1e-6)
    return hubs, authorities


def normalize_scores(scores: dict) -> dict:
    """Min-max normalize a {doc_id: score} dict to [0, 1] so it can be
    linearly combined with cosine-similarity scores."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


if __name__ == "__main__":
    from indexing.link_graph import load_graph

    graph = load_graph()
    pagerank = normalize_scores(compute_pagerank(graph))
    _, authorities = compute_hits(graph)
    authorities = normalize_scores(authorities)

    print("Top 5 by PageRank:", sorted(pagerank.items(), key=lambda x: -x[1])[:5])
    print("Top 5 by HITS authority:", sorted(authorities.items(), key=lambda x: -x[1])[:5])
