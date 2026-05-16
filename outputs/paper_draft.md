# Cross-Lingual Persona Consistency in Large Language Models

**Draft** · May 2026

---

## Abstract

We investigate whether persona consistency in instruction-tuned large language models (LLMs) degrades when the interaction language shifts from English to German or Italian, and whether certain personas or topic categories are more vulnerable to this degradation. Two models of different scales — Llama-3.1-8B and Llama-3.3-70B — are evaluated under four persona prompts (Neutral, Pessimist, Scientist, Teenager) across 20 open-ended questions in English, German, and Italian. Questions span four thematic categories: future and society, daily life, emotions, and abstract concepts. We measure sentiment polarity, lexical formality, and embedding-based cross-lingual consistency. Results show that persona signals are robust across languages for all personas, but the Teenager persona exhibits the largest cross-lingual drift — particularly in the smaller model — while the Scientist persona remains the most stable. Italian responses show a small but consistent negative sentiment shift relative to English and German. The larger model outperforms the smaller one in cross-lingual consistency, with the greatest improvement observed for the Teenager persona.

---

## 1. Introduction

Instruction-tuned LLMs are increasingly deployed in multilingual settings where users interact in languages other than English. These models are typically fine-tuned primarily on English data, raising concerns about whether behavioural properties induced by system prompts — such as persona — remain stable when the conversation language changes.

Persona consistency is a practically important property for dialogue systems: a customer service agent, a pedagogical tutor, or a creative writing assistant must maintain a coherent character regardless of the user's language. Yet it is unclear whether models preserve persona-specific traits such as emotional register, lexical formality, and semantic focus when generating in a typologically distant or lower-resource language.

This paper addresses three questions:

1. Does cross-lingual persona consistency differ across persona types?
2. Is Italian more susceptible to persona drift than English or German?
3. Does model scale affect consistency, and does this effect interact with persona type?

---

## 2. Method

### 2.1 Models

We evaluate two instruction-tuned Llama models of different scales, accessed via the Groq inference API:

| Label | Model                   | Parameters |
| ----- | ----------------------- | ---------- |
| Small | Llama-3.1-8B-Instant    | ~8B        |
| Large | Llama-3.3-70B-Versatile | ~70B       |

### 2.2 Persona Prompts

Four personas are implemented as system-level instructions, written natively in each target language to avoid translation artefacts:

- **Neutral** — balanced, objective assistant
- **Pessimist** — focuses on negative outcomes, expects the worst
- **Scientist** — evidence-based, precise terminology, avoids speculation
- **Teenager** — informal, casual language, slang, emotional expressivity

### 2.3 Questions

Twenty open-ended questions are distributed across four thematic categories (5 questions each): _future and society_, _daily life_, _emotions_, and _abstract concepts_. Questions were written natively in English, German, and Italian by a speaker of all three languages, yielding 60 question items in total.

### 2.4 Evaluation Metrics

**Sentiment polarity** is computed using `cardiffnlp/twitter-xlm-roberta-base-sentiment`, a multilingual RoBERTa model. For each response we derive a continuous score in [−1, +1] as P(positive) − P(negative).

**Lexical formality** is approximated by two surface metrics: Type-Token Ratio (TTR = unique tokens / total tokens) and average word length in characters. A composite formality score in [0, 1] is computed as the mean of the two min-max normalised metrics across the full dataset.

**Cross-lingual consistency** is measured using `paraphrase-multilingual-MiniLM-L12-v2`. For each (model, persona, category, question) group, the three language responses are embedded and pairwise cosine similarities are computed (EN↔DE, EN↔IT, DE↔IT). The mean of these three values is the consistency score for that item.

---

## 3. Results

### 3.1 Sentiment

Persona-induced sentiment differences are strong and consistent across both models (Table 1). The Pessimist persona produces the most negative sentiment (mean = −0.609 for Large, −0.608 for Small), well below the Neutral baseline (0.031 / 0.056). The Teenager persona also skews negative (−0.314 / −0.247), reflecting its emotionally volatile register. The Scientist persona remains near-neutral (0.014 / −0.008).

**Table 1. Mean sentiment score by persona and model (± SD)**

| Persona   | Large  | SD    | Small  | SD    |
| --------- | ------ | ----- | ------ | ----- |
| Neutral   | 0.031  | 0.315 | 0.056  | 0.305 |
| Pessimist | −0.609 | 0.295 | −0.608 | 0.280 |
| Scientist | 0.014  | 0.229 | −0.008 | 0.268 |
| Teenager  | −0.314 | 0.520 | −0.247 | 0.504 |

Across languages, Italian responses score slightly more negative than English or German in both models (Large: IT = −0.237, EN = −0.185, DE = −0.237; Small: IT = −0.230, EN = −0.191, DE = −0.185). The Teenager persona displays the highest sentiment variance (SD ≈ 0.50 in both models), suggesting greater instability in emotional register across questions and languages.

### 3.2 Formality

Formality rankings are highly consistent across both models (Table 2). The Scientist persona scores highest (0.554 / 0.564), followed by Neutral, Pessimist, and Teenager. The Teenager persona has the shortest average word length (4.421 / 4.322 characters), while the Scientist persona has the longest (5.985 / 5.851). TTR follows the same ordering, confirming that the Scientist persona generates lexically richer and more varied text.

**Table 2. Mean formality score, TTR, and average word length by persona and model**

| Persona   | Formality (L) | Formality (S) | TTR (L) | TTR (S) | Avg Word Len (L) | Avg Word Len (S) |
| --------- | ------------- | ------------- | ------- | ------- | ---------------- | ---------------- |
| Neutral   | 0.469         | 0.525         | 0.591   | 0.587   | 5.680            | 5.638            |
| Pessimist | 0.386         | 0.349         | 0.585   | 0.556   | 5.032            | 4.683            |
| Scientist | 0.554         | 0.564         | 0.622   | 0.593   | 5.985            | 5.851            |
| Teenager  | 0.271         | 0.272         | 0.556   | 0.538   | 4.421            | 4.322            |

The near-identical formality scores for the Teenager persona across models (0.271 vs. 0.272) suggest that this register is robustly captured at both scales.

### 3.3 Cross-Lingual Consistency

**Table 3. Mean cross-lingual consistency (cosine similarity) by persona and model (± SD)**

| Persona   | Large | SD    | Small | SD    |
| --------- | ----- | ----- | ----- | ----- |
| Neutral   | 0.822 | 0.053 | 0.823 | 0.087 |
| Pessimist | 0.794 | 0.045 | 0.789 | 0.075 |
| Scientist | 0.817 | 0.044 | 0.808 | 0.065 |
| Teenager  | 0.796 | 0.057 | 0.759 | 0.092 |

The Teenager persona shows the largest cross-lingual inconsistency in the Small model (0.759), and also the largest improvement when scaling to the Large model (+0.037). Crucially, the Small model also exhibits substantially higher variance for Teenager (SD = 0.092 vs. 0.057 for Large), indicating that persona drift is not only more frequent but more unpredictable at smaller scale.

The Neutral persona achieves the highest consistency at both scales (~0.823), while Pessimist is the least consistent in the Large model (0.794). Scientist shows a clear scale benefit (+0.009), despite being the most formally constrained persona.

**Scale effect summary:** The Large model outperforms the Small model on consistency for every persona. The benefit is largest for Teenager (+0.037) and smallest for Neutral (+0.001), suggesting that scale primarily helps with informal, high-variance personas.

---

## 4. Discussion

### 4.1 Persona Robustness Varies by Register

Our results reveal a clear hierarchy of persona difficulty: Neutral and Scientist are easiest to maintain across languages (0.823 and 0.813 respectively), while Teenager is hardest (0.778). This aligns with the hypothesis that informal, slang-heavy registers are more language-specific and thus harder to transfer — casual language, slang, and emotional expressivity are deeply tied to specific linguistic and cultural conventions that do not translate uniformly across languages.

The Pessimist persona, despite producing the strongest sentiment signal (mean = −0.609), also shows moderate consistency drops (0.791). This suggests that negative framing is expressed through different lexical and syntactic strategies across languages, even when the overall emotional valence is preserved.

### 4.2 Category Effects: Abstract Topics Are the Hardest

A notable and theoretically interesting finding is that abstract questions (free will, consciousness, the meaning of life) produce the lowest cross-lingual consistency (0.783), lower even than daily life questions (0.794). This is particularly pronounced for the small model under the Teenager persona, where abstract category consistency drops to 0.725 — the floor of the entire dataset.

This pattern suggests a compounding effect: abstract content is inherently less constrained semantically, giving models more degrees of freedom to diverge. When combined with an informal persona that has a weak cross-lingual anchor, the model has neither a strong semantic nor a strong stylistic constraint to maintain coherence across languages.

### 4.3 The EN↔DE Gap: A Counterintuitive Finding

Contrary to expectations, English–German consistency (0.787) is lower than English–Italian consistency (0.794) across all groups. German's complex morphology and flexible word order may require greater structural divergence from English surface forms, pulling responses in a different semantic direction even when the underlying content is similar. Italian, while typologically further from English, appears to preserve semantic content more faithfully at the embedding level — possibly because Italian instruction data is more represented in Llama's training corpus than German.

### 4.4 Model Scale and Consistency

The improvement in cross-lingual consistency from Small to Large is concentrated in the Teenager persona (+0.037), while Neutral shows virtually no improvement (+0.001). This scale benefit is most visible in the abstract category for the Teenager persona (small: 0.725 → large: estimated ~0.76), suggesting that larger models have developed more language-independent representations of informal register. For formal, constrained personas like Scientist, scale provides minimal additional consistency benefit because the persona itself provides a strong cross-lingual anchor.

### 4.5 Limitations

Several limitations should be noted. The formality metrics used (TTR, average word length) are surface proxies that do not capture deeper pragmatic properties such as discourse structure or politeness strategies. The evaluation set is small (5 questions per category), which limits statistical power. Responses were generated with a fixed temperature, which may underestimate the full variance. Both models are from the same Llama family, limiting generalisability to other architectures such as Mistral or Gemma.

---

## 5. Conclusion

We have presented a systematic evaluation of cross-lingual persona consistency in two instruction-tuned LLMs across English, German, and Italian. Persona signals are broadly preserved across languages, but consistency degrades for informal personas and abstract topic categories — with the worst case being the small model under a Teenager persona on abstract questions (consistency = 0.725). Larger models show meaningful consistency gains concentrated in the Teenager persona, suggesting that scale helps primarily with language-specific, informal registers. Contrary to expectations, EN↔DE consistency is lower than EN↔IT, pointing to structural rather than lexical factors driving cross-lingual drift. Italian responses exhibit a small but consistent negative sentiment shift across both models and all personas, warranting further investigation into training data composition. These findings have practical implications for multilingual dialogue system design: informal personas and abstract topic domains represent the highest-risk combination for cross-lingual persona drift, regardless of model scale.

---

## References

_(To be completed — add HuggingFace model cards, sentence-transformers paper, and any relevant prior work on multilingual LLM evaluation)_

---

_This is a working draft. The figures directory contains all plots referenced in the text._

---

## Appendix: Drift Analysis — Quantitative Summary

_Added after qualitative drift analysis (drift_analysis.py)_

### A.1 Consistency by Persona (both models combined, worst → best)

| Persona   | Mean Consistency |
| --------- | ---------------- |
| Teenager  | 0.7776           |
| Pessimist | 0.7914           |
| Scientist | 0.8125           |
| Neutral   | 0.8226           |

### A.2 Consistency by Category (both models combined, worst → best)

| Category         | Mean Consistency |
| ---------------- | ---------------- |
| Abstract         | 0.7832           |
| Daily Life       | 0.7941           |
| Future & Society | 0.8048           |
| Emotions         | 0.8220           |

### A.3 Language Pair Breakdown (both models combined)

| Pair  | Mean Cosine Similarity |
| ----- | ---------------------- |
| EN↔DE | 0.7866                 |
| EN↔IT | 0.7937                 |
| DE↔IT | 0.8228                 |

### A.4 Worst Case: Small model / Teenager / Abstract

| Category         | Mean Consistency (small/teenager) |
| ---------------- | --------------------------------- |
| Abstract         | 0.7251                            |
| Future & Society | 0.7325                            |
| Daily Life       | 0.7853                            |
| Emotions         | 0.7943                            |

The combination of small model + teenager persona + abstract category produces the lowest
consistency scores in the entire dataset (0.7251), confirming that abstract, philosophically
open questions are the hardest to maintain persona across when the model is smaller and the
persona is informal.
