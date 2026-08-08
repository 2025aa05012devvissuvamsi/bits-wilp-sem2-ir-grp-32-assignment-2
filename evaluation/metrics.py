"""
IR evaluation metrics (assignment requirement F): Precision, Recall, F1,
Precision@K, Recall@K, Average Precision (-> MAP), Reciprocal Rank
(-> MRR), NDCG@K. Implemented directly since these are ranked-retrieval
metrics over relevance sets, not something sklearn provides.

All functions take `ranked_ids` (an ordered list of retrieved doc_ids,
best first) and `relevant_ids` (a set of gold-relevant doc_ids).
"""

import math


def precision(ranked_ids: list, relevant_ids: set) -> float:
    if not ranked_ids:
        return 0.0
    hits = sum(1 for d in ranked_ids if d in relevant_ids)
    return hits / len(ranked_ids)


def recall(ranked_ids: list, relevant_ids: set) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for d in ranked_ids if d in relevant_ids)
    return hits / len(relevant_ids)


def f1_score(precision_value: float, recall_value: float) -> float:
    if precision_value + recall_value == 0:
        return 0.0
    return 2 * precision_value * recall_value / (precision_value + recall_value)


def precision_at_k(ranked_ids: list, relevant_ids: set, k: int) -> float:
    return precision(ranked_ids[:k], relevant_ids)


def recall_at_k(ranked_ids: list, relevant_ids: set, k: int) -> float:
    return recall(ranked_ids[:k], relevant_ids)


def average_precision(ranked_ids: list, relevant_ids: set) -> float:
    """Average of precision@k evaluated at each rank where a relevant doc appears."""
    if not relevant_ids:
        return 0.0
    hits = 0
    precisions = []
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            precisions.append(hits / i)
    if not precisions:
        return 0.0
    return sum(precisions) / len(relevant_ids)


def reciprocal_rank(ranked_ids: list, relevant_ids: set) -> float:
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked_ids: list, relevant_ids: set, k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_ids: list, relevant_ids: set, k: int) -> float:
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_ids, relevant_ids, k) / idcg
