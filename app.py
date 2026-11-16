import os
import streamlit as st

from src.llm import PROVIDERS, get_llm
from src.loader import SUPPORTED_TYPES, load_file
from src.nlp_tools import intent, sentiment, summarise
from src.rag import RAGEngine

st.set_page_config(page_title="IntelliAssist", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "engine" not in st.session_state:
    st.session_state.engine = None
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = []
if "summary" not in st.session_state:
    st.session_state.summary = None


@st.cache_resource(show_spinner="Loading embedding model...")
def get_cached_engine():
    return RAGEngine()


with st.sidebar:
    st.title("IntelliAssist")
    st.write("Upload documents and ask questions.")

    st.subheader("Model Settings")
    provider = st.selectbox("LLM Provider", PROVIDERS)
    default_key = os.getenv("GEMINI_API_KEY" if provider == "Gemini" else "OPENAI_API_KEY", "")
    api_key = ""
    if not provider.startswith("None"):
        api_key = st.text_input("API Key", value=default_key, type="password")

    st.subheader("Document Upload")
    uploaded_files = st.file_uploader(
        "Select PDF, TXT, or DOCX files",
        type=list(SUPPORTED_TYPES),
        accept_multiple_files=True,
    )
    top_k = st.slider("Chunks to retrieve", min_value=2, max_value=8, value=4)

    if st.button("Process Documents") and uploaded_files:
        all_docs = []
        file_names = []
        with st.spinner("Reading files..."):
            for file in uploaded_files:
                try:
                    docs = load_file(file.name, file.getvalue())
                    all_docs.extend(docs)
                    file_names.append(file.name)
                except Exception as e:
                    st.error(f"Error loading {file.name}: {e}")

        if all_docs:
            engine = get_cached_engine()
            engine.llm = get_llm(provider, api_key)
            with st.spinner("Building search index..."):
                chunk_count = engine.index(all_docs)
            st.session_state.engine = engine
            st.session_state.loaded_files = file_names
            st.session_state.summary = None
            st.session_state.messages = []
            st.success(f"Indexed {chunk_count} chunks from {len(file_names)} file(s).")
        else:
            st.warning("Could not extract any text from the uploaded files.")

    if st.session_state.loaded_files:
        st.write("Loaded files:", ", ".join(st.session_state.loaded_files))

    st.subheader("Document Summary")
    summary_len = st.slider("Summary sentences", min_value=3, max_value=10, value=5)
    if st.button("Generate Summary") and st.session_state.engine is not None:
        engine = st.session_state.engine
        engine.llm = get_llm(provider, api_key)
        with st.spinner("Creating summary..."):
            st.session_state.summary = summarise(engine.full_text(), engine.llm, summary_len)

    if st.session_state.messages:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()


st.header("Document Q&A")

if st.session_state.engine is None:
    st.info("Upload a document in the sidebar and click 'Process Documents' to start.")

if st.session_state.summary:
    with st.expander("Document Summary", expanded=True):
        st.write(st.session_state.summary)


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**[{s['n']}] {s['file']} (Page {s['page']})** - Score: {s['score']}")
            preview = s["text"][:400] + ("..." if len(s["text"]) > 400 else "")
            st.caption(preview)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "user" and "meta" in msg:
            meta = msg["meta"]
            st.caption(f"Intent: {meta['intent']} | Sentiment: {meta['sentiment']['label']} ({meta['sentiment']['score']})")
        show_sources(msg.get("sources"))


user_query = st.chat_input("Ask a question about your documents...")
if user_query and st.session_state.engine is not None:
    query_meta = {
        "intent": intent(user_query),
        "sentiment": sentiment(user_query),
    }
    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
        "meta": query_meta,
    })

    with st.chat_message("user"):
        st.markdown(user_query)
        st.caption(f"Intent: {query_meta['intent']} | Sentiment: {query_meta['sentiment']['label']} ({query_meta['sentiment']['score']})")

    engine = st.session_state.engine
    engine.llm = get_llm(provider, api_key)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            if query_meta["intent"] == "Summarisation":
                response = {
                    "answer": summarise(engine.full_text(), engine.llm, summary_len),
                    "sources": [],
                }
            else:
                chat_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                response = engine.answer(user_query, history=chat_history, k=top_k)

        st.markdown(response["answer"])
        show_sources(response["sources"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response["sources"],
    })
