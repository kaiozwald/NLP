"""
evaluate.py
-----------
Runs all evaluation metrics on a responses JSON and saves outputs.

Usage:
    python evaluate.py outputs/responses_small_*.json
    python evaluate.py outputs/responses_small_*.json --judge   # also run LLM judge
    python evaluate.py outputs/responses_small_*.json --stats   # also run bootstrap CIs
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import OUTPUT_DIR
from evaluation.sentiment   import run_sentiment
from evaluation.formality   import run_formality
from evaluation.consistency import run_consistency
from evaluation.stats       import full_breakdown, print_breakdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PERSONA_ORDER   = ["neutral", "pessimist", "scientist", "teenager"]
CATEGORY_ORDER  = ["future_society", "daily_life", "emotions", "abstract"]
LANGUAGE_ORDER  = ["en", "de", "it"]


def run_evaluation(json_path: Path, run_judge: bool = False, run_stats: bool = False):
    log.info(f"Loading: {json_path}")
    df   = pd.read_json(json_path)
    stem = json_path.stem

    log.info(f"Loaded {len(df)} responses.")

    # ── Core metrics ──────────────────────────────────────────────────────────
    log.info("=== Sentiment ===")
    df = run_sentiment(df)

    log.info("=== Formality ===")
    df = run_formality(df)

    flat_path = OUTPUT_DIR / f"eval_flat_{stem}.csv"
    df.to_csv(flat_path, index=False)
    log.info(f"Flat results -> {flat_path}")

    log.info("=== Consistency ===")
    df_cons = run_consistency(df)
    cons_path = OUTPUT_DIR / f"eval_consistency_{stem}.csv"
    df_cons.to_csv(cons_path, index=False)
    log.info(f"Consistency results -> {cons_path}")

    # ── Summary printout ──────────────────────────────────────────────────────
    _print_summary(df, df_cons)

    # ── Optional: bootstrap CIs + full breakdown ──────────────────────────────
    if run_stats:
        log.info("=== Bootstrap CIs + Full Breakdown ===")
        results = full_breakdown(df, df_cons, n_boot=1000)
        print_breakdown(results)
        stats_path = OUTPUT_DIR / f"stats_{stem}.xlsx"
        with pd.ExcelWriter(stats_path) as writer:
            for key, tbl in results.items():
                tbl.to_excel(writer, sheet_name=key[:31], index=False)
        log.info(f"Stats saved -> {stats_path}")

    # ── Optional: LLM-as-judge ────────────────────────────────────────────────
    if run_judge:
        from evaluation.llm_judge import run_judge as _run_judge
        log.info("=== LLM-as-Judge ===")
        _run_judge(json_path)

    return flat_path, cons_path


def _print_summary(df: pd.DataFrame, df_cons: pd.DataFrame):
    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)

    print("\n[Sentiment] by persona:")
    print(df.groupby("persona")["sentiment_score"].mean()
          .reindex(PERSONA_ORDER).round(3).to_string())

    print("\n[Sentiment] by language:")
    print(df.groupby("language")["sentiment_score"].mean()
          .reindex(LANGUAGE_ORDER).round(3).to_string())

    print("\n[Sentiment] by persona x language (mean):")
    print(df.groupby(["persona", "language"])["sentiment_score"]
          .mean().round(3).unstack("language")
          .reindex(PERSONA_ORDER)[LANGUAGE_ORDER].to_string())

    print("\n[Formality] by persona:")
    print(df.groupby("persona")["formality_score"].mean()
          .reindex(PERSONA_ORDER).round(3).to_string())

    print("\n[Formality] by persona x language:")
    print(df.groupby(["persona", "language"])["formality_score"]
          .mean().round(3).unstack("language")
          .reindex(PERSONA_ORDER)[LANGUAGE_ORDER].to_string())

    print("\n[Consistency] by persona:")
    print(df_cons.groupby("persona")["consistency"].mean()
          .reindex(PERSONA_ORDER).round(3).to_string())

    print("\n[Consistency] by category:")
    print(df_cons.groupby("category")["consistency"].mean()
          .reindex(CATEGORY_ORDER).round(3).to_string())

    print("\n[Consistency] by persona x category:")
    print(df_cons.groupby(["persona", "category"])["consistency"]
          .mean().round(3).unstack("category")
          .reindex(PERSONA_ORDER)[CATEGORY_ORDER].to_string())

    print("\n[Consistency] language-pair means:")
    for pair in ["cos_en_de", "cos_en_it", "cos_de_it"]:
        print(f"  {pair}: {df_cons[pair].mean():.4f}")

    print(f"\nFiles saved to outputs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=Path)
    parser.add_argument("--judge",  action="store_true", help="Run LLM-as-judge scoring")
    parser.add_argument("--stats",  action="store_true", help="Run bootstrap CIs + full breakdown")
    args = parser.parse_args()

    if not args.json_file.exists():
        print(f"File not found: {args.json_file}")
        sys.exit(1)

    run_evaluation(args.json_file, run_judge=args.judge, run_stats=args.stats)