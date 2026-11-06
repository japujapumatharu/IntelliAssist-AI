# 📄 IntelliAssist AI — Smart Document AI Assistant

Upload PDF / TXT / DOCX files and chat with them. Built with **RAG**: BERT embeddings → FAISS → hybrid
(semantic + TF-IDF) retrieval → LLM answer with **source citations**.

## Features
| Feature | Where |
|---|---|
| Document upload (PDF/TXT/DOCX) | `src/loader.py` |
| RAG chatbot (LangChain splitter + FAISS) | `src/rag.py` |
| Semantic search (all-MiniLM-L6-v2 BERT embeddings) | `src/rag.py` |
| Hybrid TF-IDF + semantic re-ranking | `src/rag.py` |
| Text summarisation (LLM map-reduce, or offline TF-IDF extractive) | `src/nlp_tools.py` |
| Sentiment (HF DistilBERT) & intent analysis | `src/nlp_tools.py` |
| Source citation display + relevance score | `app.py` |
| Conversation history + export | `app.py` |
| Streamlit web UI | `app.py` |

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
1. Pick an LLM provider (Gemini / OpenAI) and paste a key — or leave on **None** for fully offline extractive mode.
   Keys can also be set via `GEMINI_API_KEY` / `OPENAI_API_KEY` env vars.
2. Upload documents (try `sample_docs/ai_overview.txt`) → **Process documents**.
3. Ask questions, click **Summarise**, expand **Sources** to verify answers.

First run downloads two small models from Hugging Face (~90 MB embeddings, ~270 MB sentiment).

## Architecture
```
Upload → extract & clean → chunk (800 / 150 overlap) → embed (MiniLM) → FAISS index
                                                              │
Question → intent + sentiment → embed → FAISS top-k ──┐       │
                              TF-IDF keyword score ───┴─ blend 0.7 sem + 0.3 tfidf
                                                       │
                                  LLM (or extractive) answer + [n] citations
```

## Evaluation
```bash
python eval/evaluate.py sample_docs/ai_overview.txt eval/eval_set.json
```
Reports Hit@1/3/5 and MRR. Add your own documents and questions to `eval/eval_set.json` for the Evaluation Report.

## Deploy (Streamlit Community Cloud)
1. Push this repo to GitHub.
2. share.streamlit.io → New app → select repo, main file `app.py`.
3. In **Settings → Secrets** add `GEMINI_API_KEY = "..."` (or `OPENAI_API_KEY`).

## Notes / limits
- Scanned (image-only) PDFs have no extractable text; add OCR (e.g. `pytesseract`) if needed.
- The index lives in memory per session; swap FAISS for ChromaDB (`langchain-chroma`) for persistence.
