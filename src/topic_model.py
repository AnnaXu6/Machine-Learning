
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def build_tfidf(
    texts: pd.Series,
    max_df: float = 0.95,
    min_df: int = 2,
    ngram_range: Tuple[int, int] = (1,1)
):
    """Fit a TF-IDF vectorizer and transform texts to a sparse matrix."""
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    return X, vocab, vectorizer

def fit_nmf(X, n_components: int = 5, random_state: int = 416, init: str = "nndsvd"):
    """Fit an NMF model and return (model, doc_topic_matrix, topic_term_matrix)."""
    model = NMF(n_components=n_components, init=init, random_state=random_state)
    W = model.fit_transform(X)   # documents x topics
    H = model.components_        # topics x terms
    return model, W, H

def top_words_per_topic(H: np.ndarray, vocab: np.ndarray, top_k: int = 10) -> pd.DataFrame:
    """Return a tidy table of top-k words per topic with weights."""
    rows = []
    for t_idx, row in enumerate(H):
        idx = np.argsort(row)[::-1][:top_k]
        for rank, j in enumerate(idx, start=1):
            rows.append({
                "topic": t_idx,
                "rank": rank,
                "term": vocab[j],
                "weight": float(row[j])
            })
    return pd.DataFrame(rows)

def dominant_topics(W: np.ndarray) -> np.ndarray:
    """Return the dominant topic index for each document."""
    return np.argmax(W, axis=1)
