"""
models/inference.py
-------------------
Generates responses for all (persona x language x question) combinations
using the Groq API (free tier, no local GPU needed).

Models used:
    mistral  -> mistral-saba-24b          (Groq-hosted)
    llama    -> llama-3.3-70b-versatile   (Groq-hosted)

Usage (from project root, venv active):
    python -m models.inference --model mistral --dry-run   # test pipeline
    python -m models.inference --model mistral             # full run
    python -m models.inference --model llama               # llama run
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

# -- Project imports -----------------------------------------------------------
sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS, OUTPUT_DIR
from data.questions import (
    CATEGORIES,
    LANGUAGES,
    PERSONA_NAMES,
    QUESTIONS,
    get_prompt,
)

# -- Env + logging -------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# -- Single response generator -------------------------------------------------

def generate_response(
    client: Groq,
    messages: list[dict],
    model_id: str,
    max_new_tokens: int = 300,
    retries: int = 3,
) -> str:
    """
    Calls the Groq chat completions endpoint and returns the reply as a string.
    Retries up to `retries` times on transient errors with exponential backoff.
    """
    for attempt in range(1, retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
            )
            return completion.choices[0].message.content.strip()

        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"Attempt {attempt}/{retries} failed: {e}. Retrying in {wait}s ...")
            time.sleep(wait)

    raise RuntimeError(f"All {retries} attempts failed for messages: {messages}")


# -- Full experiment runner ----------------------------------------------------

def run_experiment(model_key: str, dry_run: bool = False) -> Path:
    """
    Iterates over all (persona x category x language x question) combinations,
    generates a response for each, and saves results to a timestamped JSON file.

    Args:
        model_key : "mistral" or "llama"
        dry_run   : skips API calls, fills placeholder responses (pipeline test)

    Returns:
        Path to the saved JSON file.
    """
    log.info(f"=== Starting experiment | model={model_key} | dry_run={dry_run} ===")

    if not dry_run:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Add it to your .env file: GROQ_API_KEY=gsk_..."
            )
        client   = Groq(api_key=GROQ_API_KEY)
        model_id = MODELS[model_key]["groq_id"]
        log.info(f"Using Groq model: {model_id}")
    else:
        client   = None
        model_id = "dry-run"
        log.info("DRY RUN -- no API calls will be made.")

    results = []
    total   = len(PERSONA_NAMES) * len(CATEGORIES) * len(LANGUAGES) * 5
    log.info(f"Total generations: {total}")

    with tqdm(total=total, desc=f"{model_key} ({MODELS[model_key]['groq_id']})") as pbar:
        for persona in PERSONA_NAMES:
            for category in CATEGORIES:
                for lang in LANGUAGES:
                    questions = QUESTIONS[category][lang]
                    for q_idx, question in enumerate(questions):

                        messages = get_prompt(persona, lang, question)

                        if dry_run:
                            response = (
                                f"[DRY RUN] persona={persona} "
                                f"lang={lang} cat={category} q={q_idx}"
                            )
                            latency = 0.0
                        else:
                            t0       = time.time()
                            response = generate_response(
                                client,
                                messages,
                                model_id,
                                max_new_tokens=MODELS[model_key]["max_new_tokens"],
                            )
                            latency  = round(time.time() - t0, 2)

                            # Groq free tier: stay within rate limits
                            time.sleep(0.5)

                        results.append({
                            "model"         : model_key,
                            "groq_model_id" : model_id,
                            "persona"       : persona,
                            "category"      : category,
                            "language"      : lang,
                            "question_index": q_idx,
                            "question"      : question,
                            "response"      : response,
                            "latency_s"     : latency,
                        })

                        pbar.update(1)

    # -- Save ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix    = "_dryrun" if dry_run else ""
    out_path  = OUTPUT_DIR / f"responses_{model_key}{suffix}_{timestamp}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    log.info(f"Saved {len(results)} records -> {out_path}")
    return out_path


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate persona-conditioned responses via Groq API."
    )
    parser.add_argument(
        "--model",
        choices=["small", "large"],
        default="small",
        help="Which model to run (default: mistral)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test the full pipeline without making any API calls",
    )
    args     = parser.parse_args()
    out_path = run_experiment(model_key=args.model, dry_run=args.dry_run)
    print(f"\nDone! Results saved to: {out_path}")