"""
Ranked search over the indexed collection (assignment requirement D).

Pipeline per query:
1. Clean (+ optionally expand) the query text.
2. Use the inverted index for cheap candidate filtering (boolean OR/AND)
   instead of scoring every document.
3. Score candidates by TF-IDF cosine similarity to the query (content score).
4. Optionally blend in a normalized link-authority score (PageRank or HITS)
   for a hybrid score: alpha * content + (1 - alpha) * link.

SearchEngine loads all index artifacts once so repeated queries (e.g. from
the Streamlit search interface) don't re-read disk every time.
"""

import os
import re

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from indexing.inverted_index import docs_matching_all, docs_matching_any, load_index
from indexing.link_graph import load_graph
from indexing.vector_index import load_vector_index
from search.query_processor import process_query
from search.ranking import compute_hits, compute_pagerank, normalize_scores

RAW_DIR = "data/raw"
META_PATH = "data/metadata/metadata.csv"

RESULT_COLUMNS = [
    "doc_id", "title", "url", "seed_source", "snippet",
    "content_score", "link_score", "final_score",
]


class SearchEngine:
    def __init__(self, index_dir: str = "data/index"):
        self.inverted_index = load_index(out_dir=index_dir)
        self.vectorizer, self.doc_matrix, self.doc_index = load_vector_index(out_dir=index_dir)
        self.doc_id_to_pos = {doc_id: i for i, doc_id in enumerate(self.doc_index["doc_id"])}

        meta = pd.read_csv(META_PATH)
        stored = meta[~meta["is_duplicate"]].set_index("doc_id")
        self.url_by_doc = stored["url"].to_dict()
        self.raw_file_by_doc = stored["raw_file"].to_dict()

        graph = load_graph()
        self.pagerank = normalize_scores(compute_pagerank(graph))
        _, authorities = compute_hits(graph)
        self.hits_authority = normalize_scores(authorities)

    def _link_scores(self, algorithm: str) -> dict:
        if algorithm == "hits":
            return self.hits_authority
        return self.pagerank

    def _snippet(self, doc_id: str, tokens: list, window: int = 180) -> str:
        raw_file = self.raw_file_by_doc.get(doc_id)
        if not raw_file:
            return ""
        with open(os.path.join(RAW_DIR, raw_file), "r", encoding="utf-8") as f:
            text = f.read()

        match_pos = None
        for token in tokens:
            m = re.search(re.escape(token), text, re.IGNORECASE)
            if m and (match_pos is None or m.start() < match_pos):
                match_pos = m.start()

        if match_pos is None:
            return text[:window].strip() + "..."
        start = max(0, match_pos - window // 2)
        return ("..." if start > 0 else "") + text[start:start + window].strip() + "..."

    def search(
        self,
        query: str,
        top_k: int = 10,
        expand: bool = False,
        boolean_mode: str = "OR",
        rank_method: str = "hybrid",
        link_algorithm: str = "pagerank",
        alpha: float = 0.7,
    ) -> pd.DataFrame:
        """
        rank_method: "content" (pure TF-IDF similarity), "link" (pure
        PageRank/HITS), or "hybrid" (alpha-weighted combination).
        boolean_mode: "OR" (any query term) or "AND" (all query terms)
        for candidate filtering.
        """
        processed = process_query(query, expand=expand)
        candidate_tokens = processed["expanded_tokens"] if expand else processed["tokens"]
        if not candidate_tokens:
            return pd.DataFrame(columns=RESULT_COLUMNS)

        if boolean_mode == "AND":
            candidate_ids = docs_matching_all(self.inverted_index, processed["tokens"])
        else:
            candidate_ids = docs_matching_any(self.inverted_index, candidate_tokens)
        if not candidate_ids:
            return pd.DataFrame(columns=RESULT_COLUMNS)

        query_text = processed["expanded_text"] if expand else processed["text"]
        q_vec = self.vectorizer.transform([query_text])

        candidate_ids = list(candidate_ids)
        positions = [self.doc_id_to_pos[d] for d in candidate_ids]
        sims = cosine_similarity(q_vec, self.doc_matrix[positions]).ravel()
        content_scores = dict(zip(candidate_ids, sims))

        link_scores_all = self._link_scores(link_algorithm)
        rows = []
        for doc_id in candidate_ids:
            content_score = float(content_scores[doc_id])
            link_score = float(link_scores_all.get(doc_id, 0.0))
            if rank_method == "content":
                final_score = content_score
            elif rank_method == "link":
                final_score = link_score
            else:
                final_score = alpha * content_score + (1 - alpha) * link_score

            doc_row = self.doc_index.loc[self.doc_id_to_pos[doc_id]]
            rows.append({
                "doc_id": doc_id,
                "title": doc_row["title"],
                "url": self.url_by_doc.get(doc_id, ""),
                "seed_source": doc_row["seed_source"],
                "snippet": self._snippet(doc_id, processed["tokens"]),
                "content_score": content_score,
                "link_score": link_score,
                "final_score": final_score,
            })

        results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
        return results.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)

    def compare_rankings(self, query: str, top_k: int = 10, alpha: float = 0.7, link_algorithm: str = "pagerank"):
        """
        Runs the same query through content-only ranking and hybrid ranking,
        to show concretely how link-based authority reorders results.
        """
        content_only = self.search(query, top_k=top_k, rank_method="content")
        hybrid = self.search(query, top_k=top_k, rank_method="hybrid", alpha=alpha, link_algorithm=link_algorithm)
        return content_only, hybrid


if __name__ == "__main__":
    engine = SearchEngine()

    for q in ["neural networks", "robots in manufacturing", "language understanding"]:
        print(f"\n=== Query: '{q}' ===")
        results = engine.search(q, top_k=5)
        print(results[["title", "seed_source", "content_score", "link_score", "final_score"]].to_string(index=False))

    print("\n=== Content-only vs Hybrid comparison for 'artificial intelligence' ===")
    content_only, hybrid = engine.compare_rankings("artificial intelligence", top_k=5)
    print("Content-only order:", content_only["title"].tolist())
    print("Hybrid order:      ", hybrid["title"].tolist())
