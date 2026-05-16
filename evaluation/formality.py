"""
evaluation/formality.py
-----------------------
Computes lexical formality proxies for each response.

Metrics (all language-agnostic, no model download needed):
    ttr         : Type-Token Ratio = unique_words / total_words
                  Higher → more varied vocabulary → proxy for formal/educated register
    avg_word_len: Average word length in characters
                  Higher → longer, more technical words → proxy for formal register
    formality_score: simple composite = 0.5 * norm(ttr) + 0.5 * norm(avg_word_len)
                     normalised per-column to [0, 1] at the DataFrame level

Note: TTR is length-sensitive (shorter texts have artificially high TTR).
      We control for this in analysis by always comparing within the same question.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))


# ── Per-response metrics ──────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercased alphabetic tokens only (works for EN / DE / IT)."""
    return re.findall(r"[a-zA-ZäöüÄÖÜßàèéìòùÀÈÉÌÒÙ]+", text.lower())


def compute_ttr(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 4)


def compute_avg_word_len(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return round(sum(len(t) for t in tokens) / len(tokens), 4)


def compute_formality_metrics(text: str) -> dict:
    return {
        "ttr"         : compute_ttr(text),
        "avg_word_len": compute_avg_word_len(text),
    }


# ── DataFrame-level runner ────────────────────────────────────────────────────

def run_formality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds ttr, avg_word_len, and formality_score columns to the DataFrame.

    formality_score is a min-max normalised composite across the full dataset,
    so it's only meaningful for relative comparisons within this experiment.

    Args:
        df: DataFrame with at least a "response" column.

    Returns:
        df copy with three new columns.
    """
    ttrs, awls = [], []
    for text in tqdm(df["response"], desc="formality"):
        m = compute_formality_metrics(str(text))
        ttrs.append(m["ttr"])
        awls.append(m["avg_word_len"])

    df = df.copy()
    df["ttr"]          = ttrs
    df["avg_word_len"] = awls

    # Min-max normalise each metric then average into a composite score
    def minmax(series: pd.Series) -> pd.Series:
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - mn) / (mx - mn)

    df["formality_score"] = (
        0.5 * minmax(df["ttr"]) + 0.5 * minmax(df["avg_word_len"])
    ).round(4)

    return df


if __name__ == "__main__":
    tests = [
        ("teenager/en", "omg this is so annoying lol i literally can't even rn"),
        ("scientist/en", "The empirical evidence suggests a statistically significant correlation between the variables."),
        ("teenager/it", "dai ma che roba è questa non ci credo proprio lol"),
        ("scientist/it", "L'analisi statistica dei dati sperimentali indica una correlazione significativa tra le variabili osservate."),
    ]
    print("Formality smoke test:")
    for label, text in tests:
        m = compute_formality_metrics(text)
        print(f"  [{label:15}] ttr={m['ttr']:.3f}  avg_word_len={m['avg_word_len']:.2f}  | {text[:55]!r}")