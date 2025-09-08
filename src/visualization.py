
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_3d_topics(W3: np.ndarray, filename: Optional[str] = None, s: int = 2, alpha: float = 0.5):
    """Scatter plot of 3D document-topic embeddings (W with 3 columns)."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(W3[:, 0], W3[:, 1], W3[:, 2], s=s, alpha=alpha)
    ax.set_xlabel("Topic 0"); ax.set_ylabel("Topic 1"); ax.set_zlabel("Topic 2")
    ax.set_title("Tweets in 3D Topic Space (NMF, k=3)")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=180, bbox_inches="tight")
    return fig

def plot_top_words_bars(top_words_df: pd.DataFrame, filename: Optional[str] = None, top_k: int = 10):
    """Bar charts of top words per topic in a single vertical figure."""
    topics = sorted(top_words_df["topic"].unique())
    n = len(topics)
    fig_height = max(2.0 * n, 4.0)
    fig, axes = plt.subplots(n, 1, figsize=(8, fig_height))
    if n == 1:
        axes = [axes]
    for ax, t in zip(axes, topics):
        sub = (top_words_df[top_words_df["topic"] == t]
               .sort_values("rank")
               .head(top_k))
        # Horizontal bars with default colors
        ax.barh(sub["term"], sub["weight"])
        ax.invert_yaxis()
        ax.set_title(f"Topic #{t}")
        ax.set_xlabel("Weight")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=180, bbox_inches="tight")
    return fig
