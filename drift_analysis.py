"""
drift_analysis.py
-----------------
Finds the worst cross-lingual persona drift cases from the consistency CSV
and prints them with the actual responses side by side for qualitative analysis.

Usage (from project root):
    python drift_analysis.py                          # uses all consistency CSVs
    python drift_analysis.py --top 10                 # show top 10 worst cases
    python drift_analysis.py --persona teenager       # filter by persona
    python drift_analysis.py --model small            # filter by model
    python drift_analysis.py --save                   # also save to outputs/drift_cases.md
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_consistency_data() -> pd.DataFrame:
    files = glob.glob("outputs/eval_consistency_responses_*.csv")
    if not files:
        print("No consistency CSVs found in outputs/. Run evaluate.py first.")
        sys.exit(1)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} consistency records from {len(files)} file(s).")
    return df


def print_case(row: pd.Series, rank: int):
    sep = "─" * 72
    print(f"\n{'═'*72}")
    print(f"  #{rank:02d}  |  model={row['model']}  persona={row['persona']}"
          f"  category={row['category']}  q={int(row['question_index'])}")
    print(f"        consistency={row['consistency']:.4f}  "
          f"(en↔de={row['cos_en_de']:.4f}  "
          f"en↔it={row['cos_en_it']:.4f}  "
          f"de↔it={row['cos_de_it']:.4f})")
    print(f"{'═'*72}")
    print(f"\n  🇬🇧 EN:\n  {row['response_en']}\n")
    print(f"{sep}")
    print(f"\n  🇩🇪 DE:\n  {row['response_de']}\n")
    print(f"{sep}")
    print(f"\n  🇮🇹 IT:\n  {row['response_it']}\n")


def format_case_md(row: pd.Series, rank: int) -> str:
    return (
        f"### #{rank:02d} — model={row['model']} | persona={row['persona']} "
        f"| category={row['category']} | q={int(row['question_index'])}\n\n"
        f"**Consistency:** {row['consistency']:.4f} "
        f"(EN↔DE={row['cos_en_de']:.4f}, "
        f"EN↔IT={row['cos_en_it']:.4f}, "
        f"DE↔IT={row['cos_de_it']:.4f})\n\n"
        f"**EN:** {row['response_en']}\n\n"
        f"**DE:** {row['response_de']}\n\n"
        f"**IT:** {row['response_it']}\n\n"
        f"---\n"
    )


def run(top: int, persona: str | None, model: str | None, save: bool):
    df = load_consistency_data()

    if persona:
        df = df[df["persona"] == persona]
        print(f"Filtered to persona='{persona}': {len(df)} records")
    if model:
        df = df[df["model"] == model]
        print(f"Filtered to model='{model}': {len(df)} records")

    worst = df.nsmallest(top, "consistency").reset_index(drop=True)

    print(f"\n{'▼'*72}")
    print(f"  TOP {top} WORST DRIFT CASES  (lowest cross-lingual consistency)")
    print(f"{'▼'*72}")

    md_lines = [f"# Drift Cases — Top {top} Worst\n\n"]
    for i, row in worst.iterrows():
        print_case(row, i + 1)
        md_lines.append(format_case_md(row, i + 1))

    # ── Summary stats ─────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("SUMMARY: mean consistency by persona (worst → best)")
    print(
        df.groupby("persona")["consistency"]
        .mean()
        .sort_values()
        .round(4)
        .to_string()
    )

    print("\nSUMMARY: mean consistency by category (worst → best)")
    print(
        df.groupby("category")["consistency"]
        .mean()
        .sort_values()
        .round(4)
        .to_string()
    )

    print("\nSUMMARY: which language pair drops most (mean across all groups)")
    for pair in ["cos_en_de", "cos_en_it", "cos_de_it"]:
        print(f"  {pair}: {df[pair].mean():.4f}")

    if save:
        out = Path("outputs/drift_cases.md")
        out.write_text("".join(md_lines), encoding="utf-8")
        print(f"\nSaved drift cases → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qualitative drift case analysis.")
    parser.add_argument("--top",     type=int, default=8,       help="Number of worst cases to show")
    parser.add_argument("--persona", type=str, default=None,    help="Filter by persona")
    parser.add_argument("--model",   type=str, default=None,    help="Filter by model (small/large)")
    parser.add_argument("--save",    action="store_true",        help="Save cases to outputs/drift_cases.md")
    args = parser.parse_args()
    run(top=args.top, persona=args.persona, model=args.model, save=args.save)