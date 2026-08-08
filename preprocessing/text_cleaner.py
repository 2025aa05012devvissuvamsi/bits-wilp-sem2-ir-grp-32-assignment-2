"""
Configurable text cleaning / tokenization pipeline (assignment requirement C).

Exposes clean() with independently toggleable steps so different
preprocessing strategies can be compared against each other
(see comparative_analysis.py).
"""

import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

STOPWORDS = set(stopwords.words("english"))
_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()

TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")


def tokenize(text: str) -> list:
    """Lowercase + extract alphabetic tokens of length >= 2."""
    return TOKEN_RE.findall(text.lower())


def clean(text: str, remove_stopwords: bool = True, normalize: str = "lemmatize") -> list:
    """
    Run the configurable cleaning pipeline over raw text and return a token list.

    normalize: one of "lemmatize", "stem", "none"
    """
    if normalize not in ("lemmatize", "stem", "none"):
        raise ValueError(f"Unknown normalize mode: {normalize}")

    tokens = tokenize(text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    if normalize == "lemmatize":
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    elif normalize == "stem":
        tokens = [_stemmer.stem(t) for t in tokens]

    return tokens


def clean_to_string(text: str, remove_stopwords: bool = True, normalize: str = "lemmatize") -> str:
    """Convenience wrapper returning a space-joined string, e.g. for sklearn vectorizers."""
    return " ".join(clean(text, remove_stopwords=remove_stopwords, normalize=normalize))
