"""
Ground-truth relevance judgments for evaluation (assignment requirement F).

Methodology: with a 49-document corpus and no real query logs, relevance
is defined by topic match against `seed_source` (the crawl seed topic each
document was reached from) rather than per-document manual judgment. This
is a proxy, not expert relevance judgment, and should be reported as such
alongside the evaluation numbers.
"""

import pandas as pd

from indexing.vector_index import load_vector_index

TEST_QUERIES = [
    {"query": "artificial intelligence overview", "relevant_topics": ["Artificial intelligence"]},
    {"query": "machine learning algorithms", "relevant_topics": ["Machine learning"]},
    {"query": "deep learning neural networks", "relevant_topics": ["Deep learning", "Neural network"]},
    {"query": "natural language processing", "relevant_topics": ["Natural language processing"]},
    {"query": "computer vision image recognition", "relevant_topics": ["Computer vision"]},
    {"query": "robotics and automation", "relevant_topics": ["Robotics"]},
    {"query": "neural network architecture", "relevant_topics": ["Neural network"]},
    {"query": "AI applications in robotics", "relevant_topics": ["Artificial intelligence", "Robotics"]},
]


def build_relevance_sets(index_dir: str = "data/index") -> list:
    """Resolve each test query's relevant_topics into a concrete set of doc_ids."""
    _, _, doc_index = load_vector_index(out_dir=index_dir)

    judgments = []
    for item in TEST_QUERIES:
        relevant_ids = set(
            doc_index.loc[doc_index["seed_source"].isin(item["relevant_topics"]), "doc_id"]
        )
        judgments.append({
            "query": item["query"],
            "relevant_topics": item["relevant_topics"],
            "relevant_doc_ids": relevant_ids,
        })
    return judgments


if __name__ == "__main__":
    for j in build_relevance_sets():
        print(f"{j['query']!r} -> {len(j['relevant_doc_ids'])} relevant docs ({j['relevant_topics']})")
