"""
evaluation/llm_judge.py
-----------------------
LLM-as-judge evaluation of persona adherence using the Groq API.

For each response, Claude/Llama is asked to score it on three dimensions:
    - persona_adherence : does the response match the intended persona? [1-5]
    - fluency           : is the response fluent and natural in its language? [1-5]
    - drift             : does the response show signs of persona drift? [1-5]
                          (1 = strong drift, 5 = no drift at all)

The judge prompt is always in English regardless of the response language,
to keep the evaluation consistent. The judge sees: persona description,
target language, and the actual response.

Usage:
    python -m evaluation.llm_judge outputs/responses_small_*.json
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.questions import PERSONAS

load_dotenv()
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

JUDGE_MODEL = "llama-3.3-70b-versatile"

# ── Persona descriptions for the judge (always in English) ────────────────────
PERSONA_DESC = {
    "neutral"   : "a balanced, objective assistant that answers thoughtfully",
    "pessimist" : "a deeply pessimistic person who focuses on negatives and expects the worst",
    "scientist" : "a rigorous scientist who uses precise terminology and avoids speculation",
    "teenager"  : "a 16-year-old using casual informal language, slang, and strong emotions",
}

JUDGE_PROMPT = """\
You are an expert evaluator assessing whether a language model response matches a given persona.

PERSONA: {persona_desc}
RESPONSE LANGUAGE: {language}
RESPONSE:
\"\"\"
{response}
\"\"\"

Rate the response on three dimensions. Respond ONLY with a JSON object, no explanation:
{{
  "persona_adherence": <int 1-5>,
  "fluency": <int 1-5>,
  "drift": <int 1-5>
}}

Scoring:
- persona_adherence: 1=completely ignores persona, 5=perfectly embodies persona
- fluency: 1=unnatural/broken language, 5=perfectly natural for the target language
- drift: 1=strong persona drift (breaks character), 5=no drift at all (stays in character)
"""


def judge_response(
    client: Groq,
    persona: str,
    language: str,
    response: str,
    retries: int = 3,
) -> dict:
    prompt = JUDGE_PROMPT.format(
        persona_desc=PERSONA_DESC[persona],
        language=language,
        response=response[:800],  # truncate very long responses
    )
    for attempt in range(1, retries + 1):
        try:
            completion = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.0,   # deterministic judge
            )
            raw  = completion.choices[0].message.content.strip()
            # Strip markdown fences if present
            raw  = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            return {
                "judge_persona_adherence": int(data.get("persona_adherence", -1)),
                "judge_fluency"          : int(data.get("fluency", -1)),
                "judge_drift"            : int(data.get("drift", -1)),
            }
        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"Judge attempt {attempt}/{retries} failed: {e}. Retry in {wait}s")
            time.sleep(wait)

    return {"judge_persona_adherence": -1, "judge_fluency": -1, "judge_drift": -1}


def run_judge(json_path: Path) -> Path:
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    client = Groq(api_key=groq_key)

    log.info(f"Loading responses from {json_path}")
    with open(json_path) as f:
        records = json.load(f)

    log.info(f"Judging {len(records)} responses with {JUDGE_MODEL} ...")
    for rec in tqdm(records, desc="judging"):
        scores = judge_response(
            client,
            persona=rec["persona"],
            language=rec["language"],
            response=rec["response"],
        )
        rec.update(scores)
        time.sleep(0.4)  # rate limit

    # Save enriched JSON
    out_path = json_path.parent / json_path.name.replace(
        "responses_", "responses_judged_"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    log.info(f"Saved judged responses -> {out_path}")

    # Print quick summary
    df = pd.DataFrame(records)
    print("\n=== JUDGE SUMMARY ===")
    print("\nMean persona_adherence by persona:")
    print(df.groupby("persona")["judge_persona_adherence"].mean().round(2).to_string())
    print("\nMean drift score by persona (5=no drift, 1=heavy drift):")
    print(df.groupby("persona")["judge_drift"].mean().round(2).to_string())
    print("\nMean fluency by language:")
    print(df.groupby("language")["judge_fluency"].mean().round(2).to_string())
    print("\nMean drift by language:")
    print(df.groupby("language")["judge_drift"].mean().round(2).to_string())

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-judge persona evaluation.")
    parser.add_argument(
        "json_file",
        type=Path,
        nargs="?",
        help="Path to responses JSON. If omitted, uses most recent in outputs/",
    )
    args = parser.parse_args()

    if args.json_file:
        path = args.json_file
    else:
        files = sorted(glob.glob("outputs/responses_*.json"))
        files = [f for f in files if "judged" not in f]
        if not files:
            print("No response JSON found. Run models/inference.py first.")
            sys.exit(1)
        path = Path(files[-1])
        print(f"Using most recent: {path}")

    run_judge(path)