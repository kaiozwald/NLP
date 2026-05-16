"""
evaluation/sentiment.py
-----------------------
Computes sentiment polarity for each response using a multilingual model.

Model: cardiffnlp/twitter-xlm-roberta-base-sentiment
  - Supports EN, DE, IT natively
  - Returns: negative / neutral / positive + confidence scores
  - Runs on CPU, ~500MB download, cached after first run

Output columns added to DataFrame:
    sentiment_label  : "positive" | "neutral" | "negative"
    sentiment_score  : float in [-1, +1]
                       (P(positive) - P(negative), so negative values = negative sentiment)
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

sys.path.append(str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)

SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MAX_CHARS = 512  # truncate long responses to avoid tokenizer overflow


def load_sentiment_pipeline():
    log.info(f"Loading sentiment model: {SENTIMENT_MODEL}")
    return pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        top_k=None,      # return all labels + scores
        truncation=True,
        max_length=128,
    )


def score_text(pipe, text: str) -> tuple[str, float]:
    """
    Returns (label, score) where score in [-1, +1].
    score = P(positive) - P(negative)
    """
    text    = text[:MAX_CHARS]
    results = pipe(text)[0]   # list of {label, score}
    scores  = {r["label"].lower(): r["score"] for r in results}

    pos   = scores.get("positive", 0.0)
    neg   = scores.get("negative", 0.0)
    label = max(scores, key=scores.get)

    return label, round(pos - neg, 4)


def run_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds sentiment_label and sentiment_score columns to the responses DataFrame.

    Args:
        df: DataFrame with at least a "response" column.

    Returns:
        df copy with two new columns.
    """
    pipe = load_sentiment_pipeline()

    labels, scores = [], []
    for text in tqdm(df["response"], desc="sentiment"):
        label, score = score_text(pipe, str(text))
        labels.append(label)
        scores.append(score)

    df = df.copy()
    df["sentiment_label"] = labels
    df["sentiment_score"] = scores
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipe  = load_sentiment_pipeline()
    tests = [
        ("en", "Everything is terrible and there is no hope left."),
        ("en", "What a wonderful and exciting day!"),
        ("it", "La situazione è assolutamente disastrosa."),
        ("de", "Das ist wirklich wunderschön!"),
    ]
    print("\nSentiment smoke test:")
    for lang, text in tests:
        label, score = score_text(pipe, text)
        print(f"  [{lang}] {text[:55]!r:57} → {label:8} ({score:+.3f})")