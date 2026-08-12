# IR Assignment 2 — Information Retrieval System

A small end-to-end information retrieval system: it crawls Wikipedia,
cleans and studies the text, builds a search index, ranks and recommends
documents, and evaluates how well all of that works — all through a
Streamlit app.

**Live app:** https://bits-wilp-sem2-ir-grp-32-assignment-2-csmclagpjovo3yzjsj9lkr.streamlit.app/

## Running it yourself

**1. Install dependencies**

```
python -m venv venv
venv\Scripts\activate        (Windows)
source venv/bin/activate     (Mac/Linux)

pip install -r requirements.txt
```

**2. One-time download of NLTK data** (needed for stopwords/lemmatization)

```
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

**3. Start the app**

```
streamlit run app/app.py
```

That opens the Dashboard. Everything else — crawling, indexing, search,
ranking, recommendations, and evaluation — is done from the pages in the
sidebar; nothing else needs to be run separately.

If this is a fresh checkout and there's no data yet, the Dashboard will
tell you to run a crawl from the **Crawler** page first — that also
builds the search index automatically.

## What's in the project

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
