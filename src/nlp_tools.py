import re
from functools import lru_cache
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .llm import ask


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if len(p.strip()) > 25]


def extractive_summary(text, num_sentences=5):
    sents = split_sentences(text)
    if len(sents) <= num_sentences:
        return " ".join(sents)

    vec = TfidfVectorizer(stop_words="english")
    try:
        matrix = vec.fit_transform(sents)
    except ValueError:
        return " ".join(sents[:num_sentences])

    word_counts = [len(s.split()) for s in sents]
    scores = np.asarray(matrix.sum(axis=1)).ravel() / np.sqrt(word_counts)
    top_indices = sorted(np.argsort(scores)[-num_sentences:])
    return " ".join(sents[i] for i in top_indices)


def llm_summary(llm, text, num_sentences=5):
    chunk_size = 4000
    max_chunks = 8
    slices = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    if len(slices) > max_chunks:
        indices = np.linspace(0, len(slices) - 1, max_chunks).astype(int)
        slices = [slices[i] for i in indices]

    notes = [ask(llm, f"Summarize this section in 2-3 sentences:\n\n{s}") for s in slices]
    combined = "\n".join(notes)
    return ask(llm, f"Combine these points into a single summary of about {num_sentences} sentences:\n\n{combined}")


def summarise(text, llm=None, num_sentences=5):
    if llm is not None:
        try:
            return llm_summary(llm, text, num_sentences)
        except Exception:
            pass
    return extractive_summary(text, num_sentences)


@lru_cache(maxsize=1)
def get_sentiment_model():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


def sentiment(text):
    try:
        res = get_sentiment_model()(text[:512])[0]
        return {"label": res["label"].title(), "score": round(float(res["score"]), 3)}
    except Exception:
        return {"label": "Neutral", "score": 0.0}


INTENT_PATTERNS = [
    ("Greeting", r"^\s*(hi|hello|hey|good (morning|evening))\b"),
    ("Summarisation", r"\b(summar\w*|tl;?dr|overview|gist|brief|key points?)\b"),
    ("Comparison", r"\b(compare|difference|versus|vs\.?|better than)\b"),
    ("Definition", r"\b(what is|what are|define|meaning of|explain)\b"),
    ("How-to", r"\b(how (do|to|can)|steps?|procedure|process)\b"),
    ("Listing", r"\b(list|enumerate|name (all|the)|give me .* examples?)\b"),
    ("Reasoning", r"\b(why|reason|cause|because)\b"),
]


def intent(text):
    lowered = text.lower()
    for label, pattern in INTENT_PATTERNS:
        if re.search(pattern, lowered):
            return label
    return "General Question"
