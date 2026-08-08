# IR Assignment 2 — Information Retrieval System

An end-to-end IR pipeline over a Wikipedia-crawled corpus: crawling ->
text preprocessing/mining -> indexing -> search & ranking (PageRank/HITS)
-> recommendation -> evaluation, wired into a Streamlit application.

## 1. Install dependencies

```
python -m venv venv
venv\Scripts\activate        (Windows)
source venv/bin/activate     (Mac/Linux)

pip install -r requirements.txt
```

First run also needs a couple of NLTK corpora (one-time download):

```
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## 2. Run the app

From the **project root**:

```
streamlit run app/app.py
```

This opens the Dashboard. Everything else — crawling, indexing, search,
ranking, recommendations, and evaluation — is driven from the sidebar
pages; no separate scripts need to be run manually.

If `data/metadata/metadata.csv` doesn't exist yet (fresh checkout), the
Dashboard will prompt you to run a crawl from the **Crawler** page first;
that action also builds the index automatically.

## Project layout

```
crawler/        Wikipedia crawler (multi-seed, round-robin BFS, retry/backoff, dedup)
preprocessing/  Cleaning, feature engineering, keyword extraction, profiling,
                classification, comparative analysis of preprocessing strategies
indexing/       Inverted index, TF-IDF vector index, intra-corpus link graph
search/         Query processing, PageRank/HITS ranking, hybrid search engine
recommender/    Content-based, graph-based, and hybrid recommendation
evaluation/     Relevance judgments, IR metrics (P/R/F1/P@K/R@K/MAP/MRR/NDCG),
                ranking-method comparison
app/            Streamlit application (Dashboard + 7 pages)
utils/          Shared Streamlit caching/loading helpers
data/           raw/ (crawled text), metadata/ (crawl metadata + link graph),
                processed/ (preprocessing outputs), index/ (search index
                artifacts), evaluation/ (evaluation results)
```

Each module's `__main__` block can also be run standalone for development,
e.g. `python -m indexing.index_manager`, but the Streamlit app is the
intended way to exercise the whole system.
