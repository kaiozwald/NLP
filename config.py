"""
config.py
---------
Central configuration for the Cross-Lingual Persona Consistency project.
"""

from pathlib import Path

# -- Project root --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()

# -- Output directory ----------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# -- Models (via Groq API — free tier, no local download needed) --------------
#
#   small  → llama-3.1-8b-instant      fast, free, good for prototyping
#   large  → llama-3.3-70b-versatile   high quality, free, the main experiment
#
#   These are the two models we compare in the study (different scales),
#   matching the spirit of the original Mistral-7B vs Llama-3-70B proposal.
#
MODELS = {
    "small": {
        "groq_id"       : "llama-3.1-8b-instant",
        "max_new_tokens": 300,
        "label"         : "Llama-3.1-8B",
    },
    "large": {
        "groq_id"       : "llama-3.3-70b-versatile",
        "max_new_tokens": 300,
        "label"         : "Llama-3.3-70B",
    },
}

# -- Experiment parameters -----------------------------------------------------
LANGUAGES     = ["en", "de", "it"]
CATEGORIES    = ["future_society", "daily_life", "emotions", "abstract"]
PERSONA_NAMES = ["neutral", "pessimist", "teenager", "scientist"]

# -- Logging -------------------------------------------------------------------
LOG_LEVEL = "INFO"

# -- Sanity check --------------------------------------------------------------
if __name__ == "__main__":
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output dir   : {OUTPUT_DIR}")
    for k, v in MODELS.items():
        print(f"  {k}: {v['groq_id']}  ({v['label']})")