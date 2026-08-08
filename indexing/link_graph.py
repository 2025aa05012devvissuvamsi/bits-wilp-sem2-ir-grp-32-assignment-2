"""
Intra-corpus link graph, restricted to edges between documents we actually
crawled (the raw links.csv from the crawler includes ~14k edges to pages
outside the corpus, which carry no ranking signal since they're never
candidates in search results).

Consumed by search/ranking.py for PageRank / HITS (assignment requirement D).
"""

import os

import networkx as nx
import pandas as pd

LINKS_PATH = "data/metadata/links.csv"
META_PATH = "data/metadata/metadata.csv"
INDEX_DIR = "data/index"


def build_link_graph(links_path: str = LINKS_PATH, meta_path: str = META_PATH) -> nx.DiGraph:
    meta = pd.read_csv(meta_path)
    stored = meta[~meta["is_duplicate"]]
    title_to_doc_id = dict(zip(stored["title"], stored["doc_id"]))
    corpus_titles = set(title_to_doc_id.keys())

    links = pd.read_csv(links_path)
    intra = links[links["from_title"].isin(corpus_titles) & links["to_title"].isin(corpus_titles)]

    graph = nx.DiGraph()
    for doc_id in title_to_doc_id.values():
        graph.add_node(doc_id)
    for _, row in intra.iterrows():
        graph.add_edge(title_to_doc_id[row["from_title"]], title_to_doc_id[row["to_title"]])

    return graph


def save_graph(graph: nx.DiGraph, out_dir: str = INDEX_DIR, name: str = "link_graph"):
    os.makedirs(out_dir, exist_ok=True)
    nx.write_graphml(graph, os.path.join(out_dir, f"{name}.graphml"))


def load_graph(out_dir: str = INDEX_DIR, name: str = "link_graph") -> nx.DiGraph:
    return nx.read_graphml(os.path.join(out_dir, f"{name}.graphml"))


if __name__ == "__main__":
    graph = build_link_graph()
    save_graph(graph)
    print(f"Link graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} intra-corpus edges")
    isolated = [n for n in graph.nodes if graph.degree(n) == 0]
    print(f"Isolated (no intra-corpus links) documents: {len(isolated)}")
