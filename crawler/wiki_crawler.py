"""
Wikipedia Crawler for the IR Assignment.

Features (per assignment requirement B):
- Multiple seed sources
- Configurable crawl depth
- Duplicate URL handling (normalized-title dedup)
- Duplicate document handling (exact hash + near-duplicate Jaccard check)
- Metadata stored SEPARATELY from document content
"""

import hashlib
import json
import os
import re
import time
from collections import deque
from datetime import datetime, timezone

import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "IR-Assignment-Crawler/1.0 (educational project)"

DEFAULT_SEEDS = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Neural network",
    "Natural language processing",
    "Computer vision",
    "Robotics",
]


def _normalize_title(title: str) -> str:
    """Normalize a Wikipedia title for dedup comparisons."""
    return title.strip().replace("_", " ").lower()


def _safe_filename(title: str) -> str:
    """Turn a title into a filesystem-safe filename."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_")
    return s[:120]


def _tokenize_set(text: str) -> set:
    """Cheap tokenization into a lowercase word set, used only for near-dup Jaccard check."""
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return set(words)


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class WikiCrawler:
    def __init__(
        self,
        seeds,
        max_depth=1,
        max_pages=150,
        near_dup_threshold=0.85,
        raw_dir="data/raw",
        meta_dir="data/metadata",
        request_delay=0.5,
        session=None,
    ):
        self.seeds = list(seeds)
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.near_dup_threshold = near_dup_threshold
        self.raw_dir = raw_dir
        self.meta_dir = meta_dir
        self.request_delay = request_delay
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)

        # dedup / bookkeeping state
        self.visited_titles = set()          # normalized titles already processed (URL-level dedup)
        self.content_hashes = {}             # hash -> doc_id (exact duplicate detection)
        self.doc_token_sets = {}             # doc_id -> token set (for near-dup Jaccard checks)
        self.metadata_rows = []
        self.link_rows = []
        self.doc_counter = 0

    # ------------------------------------------------------------------
    # Wikipedia API helpers
    # ------------------------------------------------------------------
    def _fetch_page(self, title: str, max_retries=5):
        """Fetch extract text, links, and categories for a single page via one API call.

        Retries with exponential backoff on 429/5xx, honoring the Retry-After
        header when the server sends one (Wikipedia throttles aggressively
        under sustained request bursts).
        """
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|links|categories",
            "explaintext": 1,
            "exlimit": 1,
            "plnamespace": 0,   # only article-namespace links
            "pllimit": "max",
            "cllimit": "max",
            "redirects": 1,
            "titles": title,
        }
        backoff = 1.0
        for attempt in range(max_retries + 1):
            resp = self.session.get(WIKI_API, params=params, timeout=15)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == max_retries:
                    resp.raise_for_status()
                wait = float(resp.headers.get("Retry-After", backoff))
                time.sleep(wait)
                backoff *= 2
                continue
            resp.raise_for_status()
            break
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None
        page = next(iter(pages.values()))
        if "missing" in page:
            return None

        canonical_title = page.get("title", title)
        extract = page.get("extract", "") or ""
        links = [l["title"] for l in page.get("links", [])] if "links" in page else []
        categories = [c["title"] for c in page.get("categories", [])] if "categories" in page else []
        return {
            "title": canonical_title,
            "extract": extract,
            "links": links,
            "categories": categories,
        }

    # ------------------------------------------------------------------
    # Core crawl loop
    # ------------------------------------------------------------------
    def crawl(self, progress_callback=None):
        """
        BFS crawl starting from self.seeds up to self.max_depth.
        Each seed gets its own queue and pages are pulled round-robin across
        seeds, so a high-branching seed can't starve the others out of the
        max_pages budget before they're ever visited.
        progress_callback(msg: str) is optionally called for live UI updates (e.g. in Streamlit).
        """
        # Only clear files this crawler owns — leave externally-ingested
        # dataset files (see crawler/dataset_loader.py) untouched.
        for fname in os.listdir(self.raw_dir):
            if not fname.endswith("__dataset.txt"):
                os.remove(os.path.join(self.raw_dir, fname))

        seed_queues = {s: deque([(s, 0, s)]) for s in self.seeds}  # (title, depth, seed_source)
        rotation = deque(self.seeds)

        while rotation and self.doc_counter < self.max_pages:
            seed_source = rotation[0]
            q = seed_queues[seed_source]
            if not q:
                rotation.popleft()
                continue
            rotation.rotate(-1)

            title, depth, seed_source = q.popleft()
            norm = _normalize_title(title)

            # --- duplicate URL handling ---
            if norm in self.visited_titles:
                continue
            self.visited_titles.add(norm)

            try:
                page = self._fetch_page(title)
            except requests.RequestException as e:
                if progress_callback:
                    progress_callback(f"Failed to fetch '{title}': {e}")
                continue

            time.sleep(self.request_delay)

            if page is None or not page["extract"].strip():
                continue

            content = page["extract"]
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            token_set = _tokenize_set(content)

            # --- duplicate document handling: exact duplicate ---
            is_exact_dup = content_hash in self.content_hashes
            duplicate_of = self.content_hashes.get(content_hash)

            # --- duplicate document handling: near-duplicate (Jaccard on token sets) ---
            is_near_dup = False
            if not is_exact_dup:
                for existing_id, existing_tokens in self.doc_token_sets.items():
                    sim = _jaccard(token_set, existing_tokens)
                    if sim >= self.near_dup_threshold:
                        is_near_dup = True
                        duplicate_of = existing_id
                        break

            self.doc_counter += 1
            doc_id = f"doc_{self.doc_counter:05d}"
            is_duplicate = is_exact_dup or is_near_dup

            # store raw content only for non-duplicates (duplicates are logged, not stored twice)
            if not is_duplicate:
                self.content_hashes[content_hash] = doc_id
                self.doc_token_sets[doc_id] = token_set
                fname = f"{doc_id}__{_safe_filename(page['title'])}.txt"
                with open(os.path.join(self.raw_dir, fname), "w", encoding="utf-8") as f:
                    f.write(content)
            else:
                fname = None

            self.metadata_rows.append({
                "doc_id": doc_id,
                "title": page["title"],
                "url": "https://en.wikipedia.org/wiki/" + page["title"].replace(" ", "_"),
                "seed_source": seed_source,
                "depth": depth,
                "crawl_timestamp": datetime.now(timezone.utc).isoformat(),
                "content_hash": content_hash,
                "is_duplicate": is_duplicate,
                "duplicate_type": "exact" if is_exact_dup else ("near" if is_near_dup else "none"),
                "duplicate_of": duplicate_of if is_duplicate else "",
                "raw_file": fname or "",
                "num_outlinks": len(page["links"]),
                "categories": "; ".join(page["categories"][:10]),
                "extract_length_chars": len(content),
            })

            if progress_callback:
                status = "DUPLICATE" if is_duplicate else "stored"
                progress_callback(f"[{self.doc_counter}/{self.max_pages}] {page['title']} ({status})")

            # record link graph edges (for PageRank later) regardless of duplicate status
            for linked_title in page["links"]:
                self.link_rows.append({"from_title": page["title"], "to_title": linked_title})

            # enqueue next depth level
            if depth < self.max_depth:
                for linked_title in page["links"]:
                    if _normalize_title(linked_title) not in self.visited_titles:
                        seed_queues[seed_source].append((linked_title, depth + 1, seed_source))

        self._save_metadata()
        return {
            "total_processed": self.doc_counter,
            "stored": sum(1 for r in self.metadata_rows if not r["is_duplicate"]),
            "duplicates": sum(1 for r in self.metadata_rows if r["is_duplicate"]),
        }

    # ------------------------------------------------------------------
    def _save_metadata(self):
        import csv

        meta_path = os.path.join(self.meta_dir, "metadata.csv")
        if self.metadata_rows:
            # Preserve rows from externally-ingested datasets (they aren't
            # produced by this crawl and would otherwise be wiped out).
            preserved_rows = []
            if os.path.exists(meta_path):
                with open(meta_path, "r", newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if str(row.get("raw_file", "")).endswith("__dataset.txt"):
                            preserved_rows.append(row)

            keys = list(self.metadata_rows[0].keys())
            all_rows = self.metadata_rows + [
                {k: row.get(k, "") for k in keys} for row in preserved_rows
            ]
            with open(meta_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(all_rows)

        links_path = os.path.join(self.meta_dir, "links.csv")
        if self.link_rows:
            with open(links_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["from_title", "to_title"])
                writer.writeheader()
                writer.writerows(self.link_rows)

        # also dump crawl config/summary as JSON for reproducibility
        summary_path = os.path.join(self.meta_dir, "crawl_config.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "seeds": self.seeds,
                "max_depth": self.max_depth,
                "max_pages": self.max_pages,
                "near_dup_threshold": self.near_dup_threshold,
                "total_processed": self.doc_counter,
            }, f, indent=2)


if __name__ == "__main__":
    # Quick manual test hook (run this file directly with internet access to test live)
    crawler = WikiCrawler(seeds=DEFAULT_SEEDS, max_depth=1, max_pages=50)
    result = crawler.crawl(progress_callback=print)
    print("Crawl summary:", result)