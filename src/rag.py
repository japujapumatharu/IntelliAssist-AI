import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer

from .llm import ask
from .nlp_tools import split_sentences

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

QA_PROMPT = """Answer the user's question using only the provided context.
If the context does not contain the answer, state that it was not found in the documents.
Reference sources using bracket numbers like [1] or [2].

Recent chat history:
{history}

Context:
{context}

Question: {question}
Answer:"""


class RAGEngine:
    def __init__(self, llm=None, chunk_size=800, chunk_overlap=150):
        self.llm = llm
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True}
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.chunks = []
        self.vector_store = None
        self.tfidf = None
        self.tfidf_matrix = None

    def index(self, docs):
        self.chunks = self.splitter.split_documents(docs)
        for idx, chunk in enumerate(self.chunks):
            chunk.metadata["chunk_id"] = idx

        self.vector_store = FAISS.from_documents(self.chunks, self.embeddings)
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf.fit_transform([c.page_content for c in self.chunks])
        return len(self.chunks)

    def full_text(self):
        return "\n\n".join(c.page_content for c in self.chunks)

    def retrieve(self, query, k=4, semantic_weight=0.7):
        pool_size = min(max(k * 3, 10), len(self.chunks))
        faiss_hits = self.vector_store.similarity_search_with_score(query, k=pool_size)
        query_tfidf = self.tfidf.transform([query])

        scored_results = []
        for doc, l2_dist in faiss_hits:
            sem_score = max(0.0, 1.0 - (float(l2_dist) ** 2) / 2.0)
            chunk_idx = doc.metadata["chunk_id"]
            kw_score = float((self.tfidf_matrix[chunk_idx] @ query_tfidf.T).toarray()[0, 0])
            combined = semantic_weight * sem_score + (1.0 - semantic_weight) * kw_score
            scored_results.append((doc, combined))

        scored_results.sort(key=lambda item: item[1], reverse=True)
        return scored_results[:k]

    def answer(self, question, history=None, k=4):
        if self.vector_store is None:
            return {"answer": "Please upload and process a document first.", "sources": []}

        hits = self.retrieve(question, k=k)
        sources = []
        for idx, (doc, score) in enumerate(hits, start=1):
            sources.append({
                "n": idx,
                "file": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "score": round(score, 3),
                "text": doc.page_content,
            })

        if self.llm is not None:
            try:
                context_str = "\n\n".join(
                    f"[{s['n']}] ({s['file']}, page {s['page']}) {s['text']}" for s in sources
                )
                recent_history = (history or [])[-6:]
                history_str = "\n".join(f"{m['role']}: {m['content']}" for m in recent_history) or "None"
                prompt = QA_PROMPT.format(history=history_str, context=context_str, question=question)
                response_text = ask(self.llm, prompt)
                return {"answer": response_text, "sources": sources}
            except Exception as err:
                fallback_msg = f"(API error: {type(err).__name__}, falling back to offline extraction)\n\n"
                return {"answer": fallback_msg + self.extractive_answer(question, sources), "sources": sources}

        return {"answer": self.extractive_answer(question, sources), "sources": sources}

    @staticmethod
    def extractive_answer(question, sources, max_sentences=3):
        candidate_sents = [(s["n"], sent) for s in sources for sent in split_sentences(s["text"])]
        if not candidate_sents:
            return "No relevant text found in the uploaded documents."

        vec = TfidfVectorizer(stop_words="english")
        try:
            matrix = vec.fit_transform([sent for _, sent in candidate_sents] + [question])
        except ValueError:
            return candidate_sents[0][1]

        similarities = (matrix[:-1] @ matrix[-1].T).toarray().ravel()
        best_indices = sorted(np.argsort(similarities)[-max_sentences:])
        return " ".join(f"{candidate_sents[i][1]} [{candidate_sents[i][0]}]" for i in best_indices)
