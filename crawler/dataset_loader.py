"""
Secondary heterogeneous source: ingest a public dataset (CSV) alongside
the Wikipedia crawl (assignment requirement B: heterogeneous sources).
"""

import os

import pandas as pd


def _next_doc_id_start(meta_path: str, block: int = 1000) -> int:
    """Pick a doc_id start number that won't collide with existing docs,
    so repeated ingests don't silently overwrite each other."""
    if not os.path.exists(meta_path):
        return block
    existing = pd.read_csv(meta_path)
    if existing.empty:
        return block
    max_num = existing["doc_id"].str.extract(r"(\d+)").astype(int).max().iloc[0]
    return ((max_num // block) + 1) * block


def load_dataset(
    csv_path: str,
    text_col: str,
    title_col: str,
    raw_dir: str = "data/raw",
    meta_dir: str = "data/metadata",
    source_label: str = "external_dataset",
) -> int:
    """Load an external CSV dataset into the same raw/metadata layout the
    crawler uses, so the rest of the pipeline treats it identically.
    Returns the number of documents ingested."""
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise KeyError(text_col)
    if title_col not in df.columns:
        raise KeyError(title_col)

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir, "metadata.csv")
    start_doc_id = _next_doc_id_start(meta_path)

    rows = []
    for i, row in df.iterrows():
        text = str(row[text_col]).strip()
        if not text:
            continue
        doc_id = f"doc_{start_doc_id + i:05d}"
        title = str(row[title_col])
        fname = f"{doc_id}__dataset.txt"
        with open(os.path.join(raw_dir, fname), "w", encoding="utf-8") as f:
            f.write(text)
        rows.append({
            "doc_id": doc_id,
            "title": title,
            "url": "",
            "seed_source": source_label,
            "depth": 0,
            "crawl_timestamp": "",
            "content_hash": "",
            "is_duplicate": False,
            "duplicate_type": "none",
            "duplicate_of": "",
            "raw_file": fname,
            "num_outlinks": 0,
            "categories": "",
            "extract_length_chars": len(text),
        })

    if not rows:
        return 0

    existing = pd.read_csv(meta_path) if os.path.exists(meta_path) else pd.DataFrame()
    pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).to_csv(meta_path, index=False)
    return len(rows)