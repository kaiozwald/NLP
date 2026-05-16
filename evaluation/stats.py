"""
evaluation/stats.py
-------------------
Bootstrap confidence intervals and cross-tabulated breakdowns
for all three metrics.

Usage:
    from evaluation.stats import bootstrap_ci, full_breakdown

    ci = bootstrap_ci(df, group_cols=['persona','language'], metric='sentiment_score')
    breakdown = full_breakdown(df_flat, df_cons)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    df: pd.DataFrame,
    group_cols: list[str],
    metric: str,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Computes bootstrap confidence intervals for the mean of `metric`
    within each group defined by `group_cols`.

    Args:
        df         : DataFrame containing metric column and group columns
        group_cols : columns to group by, e.g. ['persona', 'language']
        metric     : column name to compute CI for
        n_boot     : number of bootstrap resamples (default 1000)
        ci         : confidence level (default 0.95)
        seed       : random seed for reproducibility

    Returns:
        DataFrame with columns: [*group_cols, 'mean', 'ci_lower', 'ci_upper', 'n']
    """
    rng   = np.random.default_rng(seed)
    alpha = 1 - ci
    rows  = []

    for key, grp in df.groupby(group_cols):
        vals = grp[metric].dropna().values
        if len(vals) < 2:
            continue

        # Bootstrap: resample with replacement n_boot times, compute mean each time
        boot_means = np.array([
            rng.choice(vals, size=len(vals), replace=True).mean()
            for _ in range(n_boot)
        ])

        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row["mean"]     = round(float(vals.mean()), 4)
        row["ci_lower"] = round(float(np.percentile(boot_means, 100 * alpha / 2)), 4)
        row["ci_upper"] = round(float(np.percentile(boot_means, 100 * (1 - alpha / 2))), 4)
        row["ci_width"] = round(row["ci_upper"] - row["ci_lower"], 4)
        row["n"]        = len(vals)
        rows.append(row)

    return pd.DataFrame(rows)


# ── Full cross-tabulated breakdown ────────────────────────────────────────────

def full_breakdown(
    df_flat: pd.DataFrame,
    df_cons: pd.DataFrame,
    n_boot: int = 1000,
) -> dict[str, pd.DataFrame]:
    """
    Produces a complete set of cross-tabulated breakdowns with bootstrap CIs
    for all three metrics (sentiment, formality, consistency).

    Returns a dict of DataFrames:
        'sentiment_persona_lang'   : persona x language x model
        'sentiment_persona_cat'    : persona x category x model
        'formality_persona_lang'   : persona x language x model
        'formality_persona_cat'    : persona x category x model
        'consistency_persona_model': persona x model  (from cons df)
        'consistency_persona_cat'  : persona x category x model
        'consistency_lang_pair'    : language pair x persona x model
    """
    print("Computing bootstrap CIs — this takes ~30s ...")
    out = {}

    # ── Sentiment ─────────────────────────────────────────────────────────────
    out["sentiment_persona_lang"] = bootstrap_ci(
        df_flat, ["model", "persona", "language"], "sentiment_score", n_boot
    )
    out["sentiment_persona_cat"] = bootstrap_ci(
        df_flat, ["model", "persona", "category"], "sentiment_score", n_boot
    )

    # ── Formality ─────────────────────────────────────────────────────────────
    out["formality_persona_lang"] = bootstrap_ci(
        df_flat, ["model", "persona", "language"], "formality_score", n_boot
    )
    out["formality_persona_cat"] = bootstrap_ci(
        df_flat, ["model", "persona", "category"], "formality_score", n_boot
    )

    # ── Consistency ───────────────────────────────────────────────────────────
    out["consistency_persona_model"] = bootstrap_ci(
        df_cons, ["model", "persona"], "consistency", n_boot
    )
    out["consistency_persona_cat"] = bootstrap_ci(
        df_cons, ["model", "persona", "category"], "consistency", n_boot
    )

    # Language-pair breakdown: melt the three pair columns into long format
    pair_cols   = ["cos_en_de", "cos_en_it", "cos_de_it"]
    pair_labels = {"cos_en_de": "EN-DE", "cos_en_it": "EN-IT", "cos_de_it": "DE-IT"}
    df_pairs = df_cons.melt(
        id_vars=["model", "persona", "category"],
        value_vars=pair_cols,
        var_name="pair", value_name="cosine"
    )
    df_pairs["pair"] = df_pairs["pair"].map(pair_labels)
    out["consistency_lang_pair"] = bootstrap_ci(
        df_pairs, ["model", "persona", "pair"], "cosine", n_boot
    )

    return out


def print_breakdown(results: dict[str, pd.DataFrame]):
    """Pretty-prints all breakdown tables."""
    titles = {
        "sentiment_persona_lang"   : "Sentiment — Persona x Language x Model",
        "sentiment_persona_cat"    : "Sentiment — Persona x Category x Model",
        "formality_persona_lang"   : "Formality — Persona x Language x Model",
        "formality_persona_cat"    : "Formality — Persona x Category x Model",
        "consistency_persona_model": "Consistency — Persona x Model",
        "consistency_persona_cat"  : "Consistency — Persona x Category x Model",
        "consistency_lang_pair"    : "Consistency — Language Pair x Persona x Model",
    }
    for key, df in results.items():
        print(f"\n{'='*70}")
        print(f"  {titles.get(key, key)}")
        print(f"{'='*70}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    import glob, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    flat_files = glob.glob("outputs/eval_flat_responses_*.csv")
    cons_files = glob.glob("outputs/eval_consistency_responses_*.csv")

    if not flat_files or not cons_files:
        print("Run evaluate.py first to generate the CSVs.")
        sys.exit(1)

    df_flat = pd.concat([pd.read_csv(f) for f in flat_files], ignore_index=True)
    df_cons = pd.concat([pd.read_csv(f) for f in cons_files], ignore_index=True)

    results = full_breakdown(df_flat, df_cons)
    print_breakdown(results)

    # Save all tables
    out_path = Path("outputs/stats_breakdown.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        for key, df in results.items():
            df.to_excel(writer, sheet_name=key[:31], index=False)
    print(f"\nAll tables saved to {out_path}")