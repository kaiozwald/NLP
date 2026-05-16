"""
evaluation/consistency.py
-------------------------
Measures cross-lingual persona consistency using sentence embeddings.

Method:
    For each (persona, category, question_index, model) group we have 3 responses —
    one per language (en, de, it). We embed all three with a multilingual model,
    then compute pairwise cosine similarities:
        en↔de, en↔it, de↔it
    The mean of these three values is the consistency score for that question.

    High score → model gives semantically similar answers regardless of language
    Low score  → the persona "drifts" when the language changes

Model: paraphrase-multilingual-MiniLM-L12-v2
  - 50+ languages including EN, DE, IT
  - ~120MB download, runs fast on CPU
  - Perfect for semantic similarity across languages

Output columns added:
    cos_en_de   : cosine similarity between EN and DE response
    cos_en_it   : cosine similarity between EN and IT response
    cos_de_it   : cosine similarity between DE and IT response
    consistency : mean of the three pairwise similarities
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def load_embedding_model() -> SentenceTransformer:
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def run_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cross-lingual consistency scores for every
    (model, persona, category, question_index) group.

    Requires all three languages to be present for each group.
    Groups with missing languages are skipped and filled with NaN.

    Args:
        df: Full responses DataFrame (output of inference.py), with columns:
            model, persona, category, language, question_index, response

    Returns:
        New DataFrame at the group level with columns:
            model, persona, category, question_index,
            response_en, response_de, response_it,
            cos_en_de, cos_en_it, cos_de_it, consistency
    """
    model = load_embedding_model()

    group_keys = ["model", "persona", "category", "question_index"]
    groups     = df.groupby(group_keys)

    rows = []
    for key, grp in tqdm(groups, desc="consistency"):
        lang_map = grp.set_index("language")["response"].to_dict()

        # Need all three languages
        if not all(l in lang_map for l in ["en", "de", "it"]):
            log.warning(f"Skipping incomplete group: {key}")
            continue

        texts = [lang_map["en"], lang_map["de"], lang_map["it"]]
        embs  = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        cos_en_de = round(cosine_similarity(embs[0], embs[1]), 4)
        cos_en_it = round(cosine_similarity(embs[0], embs[2]), 4)
        cos_de_it = round(cosine_similarity(embs[1], embs[2]), 4)
        consistency = round(np.mean([cos_en_de, cos_en_it, cos_de_it]), 4)

        model_key, persona, category, q_idx = key
        rows.append({
            "model"          : model_key,
            "persona"        : persona,
            "category"       : category,
            "question_index" : q_idx,
            "response_en"    : lang_map["en"],
            "response_de"    : lang_map["de"],
            "response_it"    : lang_map["it"],
            "cos_en_de"      : cos_en_de,
            "cos_en_it"      : cos_en_it,
            "cos_de_it"      : cos_de_it,
            "consistency"    : consistency,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    emb_model = load_embedding_model()

    tests = [
        # Should be HIGH similarity (same meaning, different languages)
        ("en", "The future of humanity depends on our ability to cooperate."),
        ("de", "Die Zukunft der Menschheit hängt von unserer Fähigkeit zur Zusammenarbeit ab."),
        ("it", "Il futuro dell'umanità dipende dalla nostra capacità di cooperare."),
    ]
    embs = emb_model.encode([t for _, t in tests], convert_to_numpy=True)
    print("\nConsistency smoke test (should be HIGH ~0.85+):")
    print(f"  en↔de: {cosine_similarity(embs[0], embs[1]):.4f}")
    print(f"  en↔it: {cosine_similarity(embs[0], embs[2]):.4f}")
    print(f"  de↔it: {cosine_similarity(embs[1], embs[2]):.4f}")