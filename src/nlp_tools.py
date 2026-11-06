"""Classic NLP helpers: sentence splitting, TF-IDF summariser, sentiment, intent."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .llm import ask


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if len(p.strip()) > 25]


# ---------- Summarisation ----------
def extractive_summary(text: str, n_sentences: int = 5) -> str:
    """TF-IDF sentence scoring; keeps original sentence order."""
    sents = split_sentences(text)
    if len(sents) <= n_sentences:
        return " ".join(sents)
    vec = TfidfVectorizer(stop_words="english")
    try:
        m = vec.fit_transform(sents)
    except ValueError:
        return " ".join(sents[:n_sentences])
    scores = np.asarray(m.sum(axis=1)).ravel() / np.sqrt([len(s.split()) for s in sents])
    top = sorted(np.argsort(scores)[-n_sentences:])
    return " ".join(sents[i] for i in top)


def llm_summary(llm, text: str, n_sentences: int = 5) -> str:
    """Map-reduce over evenly sampled slices so long docs stay within limits."""
    size, max_slices = 4000, 8
    slices = [text[i:i + size] for i in range(0, len(text), size)]
    if len(slices) > max_slices:
        idx = np.linspace(0, len(slices) - 1, max_slices).astype(int)
        slices = [slices[i] for i in idx]
    partials = [ask(llm, f"Summarise this passage in 2-3 sentences:\n\n{s}") for s in slices]
    joined = "\n".join(partials)
    return ask(llm, f"Combine these notes into one coherent summary of about {n_sentences} sentences:\n\n{joined}")


def summarise(text: str, llm=None, n_sentences: int = 5) -> str:
    if llm is not None:
        try:
            return llm_summary(llm, text, n_sentences)
        except Exception:
            pass  # fall back to local summary
    return extractive_summary(text, n_sentences)


# ---------- Sentiment (Hugging Face) ----------
@lru_cache(maxsize=1)
def _sentiment_pipe():
    from transformers import pipeline

    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


def sentiment(text: str) -> dict:
    try:
        r = _sentiment_pipe()(text[:512])[0]
        return {"label": r["label"].title(), "score": round(float(r["score"]), 3)}
    except Exception:
        return {"label": "Unavailable", "score": 0.0}


# ---------- Intent (rule based, explainable) ----------
_INTENTS = [
    ("Greeting", r"^\s*(hi|hello|hey|good (morning|evening))\b"),
    ("Summarisation", r"\b(summar\w*|tl;?dr|overview|gist|brief|key points?)\b"),
    ("Comparison", r"\b(compare|difference|versus|vs\.?|better than)\b"),
    ("Definition", r"\b(what is|what are|define|meaning of|explain)\b"),
    ("Procedure / How-to", r"\b(how (do|to|can)|steps?|procedure|process)\b"),
    ("Listing", r"\b(list|enumerate|name (all|the)|give me .* examples?)\b"),
    ("Reasoning / Why", r"\b(why|reason|cause|because)\b"),
]


def intent(text: str) -> str:
    t = text.lower()
    for name, pat in _INTENTS:
        if re.search(pat, t):
            return name
    return "Factual lookup"
