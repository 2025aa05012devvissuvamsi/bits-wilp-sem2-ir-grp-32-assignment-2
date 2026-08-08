"""
Query processing: cleaning (reusing the same pipeline as indexing, so
queries and documents live in the same vector space) plus optional
WordNet-based query expansion as a query-optimization technique
(assignment requirement D).
"""

from nltk.corpus import wordnet as wn

from preprocessing.text_cleaner import clean

MAX_SYNONYMS_PER_TERM = 2


def clean_query(query: str) -> list:
    """Same cleaning pipeline used for documents: lowercase, stopwords out, lemmatize."""
    return clean(query)


def expand_terms(tokens: list, max_synonyms_per_term: int = MAX_SYNONYMS_PER_TERM) -> list:
    """
    Add up to N single-word WordNet synonyms per token, deduplicated.
    Keeps expansion conservative (few synonyms, no multi-word phrases) to
    avoid diluting precision.
    """
    expanded = list(tokens)
    seen = set(tokens)
    for token in tokens:
        added = 0
        for synset in wn.synsets(token):
            for lemma in synset.lemma_names():
                candidate = lemma.lower().replace("_", " ")
                if " " in candidate or candidate in seen:
                    continue
                seen.add(candidate)
                expanded.append(candidate)
                added += 1
                if added >= max_synonyms_per_term:
                    break
            if added >= max_synonyms_per_term:
                break
    return expanded


def process_query(query: str, expand: bool = False) -> dict:
    """
    Returns {"tokens": cleaned tokens, "expanded_tokens": tokens (+synonyms
    if expand=True), "text": space-joined cleaned tokens for vectorization}.
    """
    tokens = clean_query(query)
    expanded_tokens = expand_terms(tokens) if expand else list(tokens)
    return {
        "tokens": tokens,
        "expanded_tokens": expanded_tokens,
        "text": " ".join(tokens),
        "expanded_text": " ".join(expanded_tokens),
    }


if __name__ == "__main__":
    result = process_query("robots and machine vision", expand=True)
    print(result)
