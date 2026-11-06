"""RAG engine: chunk -> embed (BERT) -> FAISS -> hybrid (semantic + TF-IDF) retrieval -> answer."""
from __future__ import annotations

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer

from .llm import ask
from .nlp_tools import split_sentences

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT = """You are IntelliAssist, a document assistant. Answer the question using ONLY the context.
If the answer is not in the context, say you could not find it in the uploaded documents.
Cite sources like [1], [2] matching the numbered context blocks.

Conversation so far:
{history}

Context:
{context}

Question: {question}
Answer:"""


class RAGEngine:
    def __init__(self, llm=None, chunk_size: int = 800, chunk_overlap: int = 150):
        self.llm = llm
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True}
        )
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunks: list[Document] = []
        self.store: FAISS | None = None
        self.tfidf: TfidfVectorizer | None = None
        self.tfidf_matrix = None

    # ----- indexing -----
    def index(self, docs: list[Document]) -> int:
        self.chunks = self.splitter.split_documents(docs)
        for i, c in enumerate(self.chunks):
            c.metadata["chunk_id"] = i
        self.store = FAISS.from_documents(self.chunks, self.embeddings)
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf.fit_transform([c.page_content for c in self.chunks])
        return len(self.chunks)

    @property
    def ready(self) -> bool:
        return self.store is not None

    def full_text(self) -> str:
        return "\n\n".join(c.page_content for c in self.chunks)

    # ----- retrieval -----
    def retrieve(self, query: str, k: int = 4, w_sem: float = 0.7) -> list[tuple[Document, float]]:
        fetch_k = min(max(k * 3, 10), len(self.chunks))
        hits = self.store.similarity_search_with_score(query, k=fetch_k)  # L2 distance, normalised vecs
        q_vec = self.tfidf.transform([query])
        ranked = []
        for doc, dist in hits:
            sem = max(0.0, 1 - float(dist) / 2)  # -> cosine similarity
            kw = float((self.tfidf_matrix[doc.metadata["chunk_id"]] @ q_vec.T).toarray()[0, 0])
            ranked.append((doc, w_sem * sem + (1 - w_sem) * kw))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:k]

    # ----- answering -----
    def answer(self, question: str, history: list[dict] | None = None, k: int = 4) -> dict:
        if not self.ready:
            return {"answer": "Please upload and process a document first.", "sources": []}
        results = self.retrieve(question, k=k)
        sources = [
            {
                "n": i,
                "file": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "score": round(s, 3),
                "text": d.page_content,
            }
            for i, (d, s) in enumerate(results, start=1)
        ]
        if self.llm is not None:
            try:
                ctx = "\n\n".join(f"[{s['n']}] ({s['file']}, p.{s['page']}) {s['text']}" for s in sources)
                hist = "\n".join(f"{m['role']}: {m['content']}" for m in (history or [])[-6:]) or "(none)"
                text = ask(self.llm, PROMPT.format(history=hist, context=ctx, question=question))
                return {"answer": text, "sources": sources}
            except Exception as e:  # API error -> fall back so the app never dies
                note = f"_(LLM call failed: {type(e).__name__}. Showing extractive answer.)_\n\n"
                return {"answer": note + self._extractive(question, sources), "sources": sources}
        return {"answer": self._extractive(question, sources), "sources": sources}

    @staticmethod
    def _extractive(question: str, sources: list[dict], n: int = 3) -> str:
        sents = [(s["n"], x) for s in sources for x in split_sentences(s["text"])]
        if not sents:
            return "I could not find that in the uploaded documents."
        vec = TfidfVectorizer(stop_words="english")
        try:
            m = vec.fit_transform([x for _, x in sents] + [question])
        except ValueError:
            return sents[0][1]
        sims = (m[:-1] @ m[-1].T).toarray().ravel()
        top = sorted(np.argsort(sims)[-n:])
        return " ".join(f"{sents[i][1]} [{sents[i][0]}]" for i in top)
