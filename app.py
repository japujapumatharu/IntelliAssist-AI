"""IntelliAssist AI - Smart Document AI Assistant (Streamlit UI)."""
from __future__ import annotations

import os
from datetime import datetime

import streamlit as st

from src.llm import PROVIDERS, get_llm
from src.loader import SUPPORTED, load_file
from src.nlp_tools import intent, sentiment, summarise

st.set_page_config(page_title="IntelliAssist AI", page_icon="📄", layout="wide")

# ---------- state ----------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("engine", None)
st.session_state.setdefault("files", [])
st.session_state.setdefault("summary", None)


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~90 MB)...")
def make_engine(provider: str, api_key: str):
    from src.rag import RAGEngine

    return RAGEngine(llm=get_llm(provider, api_key))


# ---------- sidebar ----------
with st.sidebar:
    st.title("📄 IntelliAssist AI")
    st.caption("Chat with your documents using RAG.")

    st.subheader("1 · Model")
    provider = st.selectbox("LLM provider", PROVIDERS)
    env_key = os.getenv("GEMINI_API_KEY" if provider == "Gemini" else "OPENAI_API_KEY", "")
    api_key = ""
    if not provider.startswith("None"):
        api_key = st.text_input(f"{provider} API key", value=env_key, type="password")
    st.caption("No key? Runs fully offline in extractive mode.")

    st.subheader("2 · Documents")
    uploads = st.file_uploader("Upload PDF / TXT / DOCX", type=list(SUPPORTED), accept_multiple_files=True)
    top_k = st.slider("Chunks retrieved (k)", 2, 8, 4)

    if st.button("⚙️ Process documents", type="primary", use_container_width=True, disabled=not uploads):
        docs, names = [], []
        with st.spinner("Extracting text..."):
            for f in uploads:
                try:
                    docs += load_file(f.name, f.getvalue())
                    names.append(f.name)
                except Exception as e:
                    st.error(f"{f.name}: {e}")
        if docs:
            engine = make_engine(provider, api_key)
            engine.llm = get_llm(provider, api_key)
            with st.spinner("Embedding and indexing..."):
                n = engine.index(docs)
            st.session_state.update(engine=engine, files=names, summary=None, messages=[])
            st.success(f"Indexed {n} chunks from {len(names)} file(s).")
        else:
            st.warning("No readable text found (scanned PDFs need OCR).")

    if st.session_state.files:
        st.markdown("**Loaded:** " + ", ".join(st.session_state.files))

    st.subheader("3 · Tools")
    n_sent = st.slider("Summary length (sentences)", 3, 10, 5)
    if st.button("📝 Summarise documents", use_container_width=True, disabled=st.session_state.engine is None):
        eng = st.session_state.engine
        eng.llm = get_llm(provider, api_key)
        with st.spinner("Summarising..."):
            st.session_state.summary = summarise(eng.full_text(), eng.llm, n_sent)

    if st.session_state.messages:
        log = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages)
        st.download_button("⬇️ Export chat", log, f"chat_{datetime.now():%Y%m%d_%H%M}.txt", use_container_width=True)
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ---------- main ----------
st.header("Ask your documents anything")

if st.session_state.engine is None:
    st.info("👈 Upload one or more documents and click **Process documents** to begin. "
            "Try `sample_docs/ai_overview.txt` first.")

if st.session_state.summary:
    with st.expander("📝 Document summary", expanded=True):
        st.write(st.session_state.summary)


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**[{s['n']}] {s['file']} · page {s['page']}** — relevance `{s['score']}`")
            st.caption(s["text"][:500] + ("…" if len(s["text"]) > 500 else ""))


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "user" and m.get("meta"):
            st.caption(f"Intent: {m['meta']['intent']} · Sentiment: {m['meta']['sentiment']['label']} "
                       f"({m['meta']['sentiment']['score']})")
        render_sources(m.get("sources"))

if prompt := st.chat_input("Ask a question about your documents...", disabled=st.session_state.engine is None):
    meta = {"intent": intent(prompt), "sentiment": sentiment(prompt)}
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": meta})
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(f"Intent: {meta['intent']} · Sentiment: {meta['sentiment']['label']} ({meta['sentiment']['score']})")

    eng = st.session_state.engine
    eng.llm = get_llm(provider, api_key)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if meta["intent"] == "Summarisation":
                result = {"answer": summarise(eng.full_text(), eng.llm, n_sent), "sources": []}
            else:
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                result = eng.answer(prompt, history, k=top_k)
        st.markdown(result["answer"])
        render_sources(result["sources"])
    st.session_state.messages.append({"role": "assistant", **result})
