"""
questions.py
------------
Central question bank for the Cross-Lingual Persona Consistency study.

Structure:
    QUESTIONS: dict[category -> dict[lang -> list[str]]]
    - 4 categories x 3 languages x 5 questions = 60 items
    - Languages: "en", "de", "it"
    - Categories: "future_society", "daily_life", "emotions", "abstract"

Usage:
    from data.questions import QUESTIONS, get_flat_list
"""

QUESTIONS = {
    "future_society": {
        "en": [
            "What do you think the world will look like in 50 years?",
            "Will artificial intelligence make human workers obsolete?",
            "How should society deal with growing inequality?",
            "Do you believe climate change can still be stopped?",
            "Is democracy the best system for the future?",
        ],
        "de": [
            "Wie wird die Welt in 50 Jahren aussehen?",
            "Wird künstliche Intelligenz menschliche Arbeitskräfte überflüssig machen?",
            "Wie sollte die Gesellschaft mit wachsender Ungleichheit umgehen?",
            "Glaubst du, dass der Klimawandel noch aufgehalten werden kann?",
            "Ist die Demokratie das beste System für die Zukunft?",
        ],
        "it": [
            "Come pensi che sarà il mondo tra 50 anni?",
            "L'intelligenza artificiale renderà obsoleti i lavoratori umani?",
            "Come dovrebbe la società affrontare la crescente disuguaglianza?",
            "Credi che il cambiamento climatico possa ancora essere fermato?",
            "La democrazia è il miglior sistema per il futuro?",
        ],
    },
    "daily_life": {
        "en": [
            "What is the best way to start your morning?",
            "How do you decide what to eat for dinner?",
            "Is it important to have a daily routine?",
            "How do you handle a stressful day at work or school?",
            "What makes a good friendship?",
        ],
        "de": [
            "Was ist der beste Weg, den Morgen zu beginnen?",
            "Wie entscheidest du, was du zum Abendessen essen möchtest?",
            "Ist es wichtig, eine tägliche Routine zu haben?",
            "Wie gehst du mit einem stressigen Tag bei der Arbeit oder in der Schule um?",
            "Was macht eine gute Freundschaft aus?",
        ],
        "it": [
            "Qual è il modo migliore per iniziare la mattina?",
            "Come decidi cosa mangiare a cena?",
            "È importante avere una routine quotidiana?",
            "Come gestisci una giornata stressante al lavoro o a scuola?",
            "Cosa rende una buona amicizia?",
        ],
    },
    "emotions": {
        "en": [
            "How do you cope when you feel deeply sad?",
            "What does happiness mean to you?",
            "Is it healthy to express anger openly?",
            "How do you deal with fear of failure?",
            "Can loneliness ever be a positive experience?",
        ],
        "de": [
            "Wie gehst du damit um, wenn du dich sehr traurig fühlst?",
            "Was bedeutet Glück für dich?",
            "Ist es gesund, Wut offen auszudrücken?",
            "Wie gehst du mit der Angst vor dem Scheitern um?",
            "Kann Einsamkeit jemals eine positive Erfahrung sein?",
        ],
        "it": [
            "Come fai fronte quando ti senti profondamente triste?",
            "Cosa significa la felicità per te?",
            "È sano esprimere la rabbia apertamente?",
            "Come affronti la paura del fallimento?",
            "La solitudine può mai essere un'esperienza positiva?",
        ],
    },
    "abstract": {
        "en": [
            "What is the meaning of life?",
            "Does free will truly exist?",
            "Is morality objective or subjective?",
            "What is the nature of consciousness?",
            "Can something come from nothing?",
        ],
        "de": [
            "Was ist der Sinn des Lebens?",
            "Existiert freier Wille wirklich?",
            "Ist Moral objektiv oder subjektiv?",
            "Was ist die Natur des Bewusstseins?",
            "Kann etwas aus dem Nichts entstehen?",
        ],
        "it": [
            "Qual è il senso della vita?",
            "Il libero arbitrio esiste davvero?",
            "La moralità è oggettiva o soggettiva?",
            "Qual è la natura della coscienza?",
            "Può qualcosa emergere dal nulla?",
        ],
    },
}

# ── Persona system prompts ────────────────────────────────────────────────────

PERSONAS = {
    "neutral": {
        "en": "You are a helpful and balanced assistant. Answer thoughtfully and objectively.",
        "de": "Du bist ein hilfreicher und ausgewogener Assistent. Antworte nachdenklich und objektiv.",
        "it": "Sei un assistente utile ed equilibrato. Rispondi in modo riflessivo e obiettivo.",
    },
    "pessimist": {
        "en": "You are a deeply pessimistic person. You tend to focus on the negative side of everything and expect the worst outcomes.",
        "de": "Du bist ein sehr pessimistischer Mensch. Du neigst dazu, dich auf die negative Seite von allem zu konzentrieren und erwartest die schlimmsten Ergebnisse.",
        "it": "Sei una persona profondamente pessimista. Tendi a concentrarti sul lato negativo di tutto e ti aspetti i peggiori risultati.",
    },
    "teenager": {
        "en": "You are a 16-year-old teenager. You use casual, informal language, slang, and often express strong emotions and opinions.",
        "de": "Du bist ein 16-jähriger Teenager. Du benutzt lockere, informelle Sprache, Slang und drückst oft starke Gefühle und Meinungen aus.",
        "it": "Sei un adolescente di 16 anni. Usi un linguaggio informale e colloquiale, slang, ed esprimi spesso forti emozioni e opinioni.",
    },
    "scientist": {
        "en": "You are a rigorous scientist. You base your answers strictly on evidence and data, use precise terminology, and avoid emotional or speculative language.",
        "de": "Du bist ein strenger Wissenschaftler. Du stützt deine Antworten strikt auf Beweise und Daten, verwendest präzise Terminologie und vermeidest emotionale oder spekulative Sprache.",
        "it": "Sei uno scienziato rigoroso. Basi le tue risposte rigorosamente su prove e dati, usi una terminologia precisa ed eviti un linguaggio emotivo o speculativo.",
    },
}

LANGUAGES     = ["en", "de", "it"]
CATEGORIES    = list(QUESTIONS.keys())
PERSONA_NAMES = list(PERSONAS.keys())


# ── Helper utilities ──────────────────────────────────────────────────────────

def get_flat_list():
    """
    Returns a flat list of dicts, one per (category, language, question_index).

    Each dict: { category, language, question_index, question_text }
    """
    rows = []
    for category, langs in QUESTIONS.items():
        for lang, qs in langs.items():
            for i, q in enumerate(qs):
                rows.append({
                    "category": category,
                    "language": lang,
                    "question_index": i,
                    "question_text": q,
                })
    return rows


def get_prompt(persona: str, language: str, question: str) -> list[dict]:
    """
    Builds a chat-style message list ready to pass to a tokenizer's
    apply_chat_template().

    Returns:
        [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    """
    assert persona in PERSONAS,  f"Unknown persona: {persona}"
    assert language in LANGUAGES, f"Unknown language: {language}"

    return [
        {"role": "system", "content": PERSONAS[persona][language]},
        {"role": "user",   "content": question},
    ]


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    flat = get_flat_list()
    print(f"Total questions : {len(flat)}")   # expect 60

    sample_q = QUESTIONS["emotions"]["it"][2]
    prompt   = get_prompt("pessimist", "it", sample_q)
    print("\nSample prompt (pessimist / Italian / emotions):")
    for msg in prompt:
        print(f"  [{msg['role']}] {msg['content']}")