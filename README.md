# IntelliAssist

A Streamlit application for querying and summarizing PDF, TXT, and DOCX documents using Retrieval-Augmented Generation (RAG).

It combines `all-MiniLM-L6-v2` sentence embeddings (via FAISS) with TF-IDF keyword scoring for hybrid retrieval. It can run offline using extractive answers or connect to Gemini or OpenAI APIs.

## Installation

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the App

Start the Streamlit interface:

```bash
streamlit run app.py
```

1. Select an LLM provider in the sidebar (or leave it on `None (Offline)` to run locally without an API key).
2. Upload one or more documents (`sample_docs/ai_overview.txt` is included for testing) and click **Process Documents**.
3. Type questions in the chat box or click **Generate Summary** in the sidebar.

On Windows, activate the virtual environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The app is normally available at `http://localhost:8501`.

## End-to-End Code Flow

The application is implemented as a Streamlit script, so Streamlit reruns
`app.py` after most user interactions. Values that must survive those reruns
are stored in `st.session_state`.

### 1. Application startup

When `app.py` starts, it:

1. Configures the page title and wide layout.
2. Initializes session-state values for chat messages, the RAG engine, loaded
   file names, and the summary.
3. Defines `get_cached_engine()`, which caches the `RAGEngine` resource so the
   embedding model is not recreated on every Streamlit rerun.
4. Builds the sidebar and the document Q&A area.

The first indexing operation can download and load the
`sentence-transformers/all-MiniLM-L6-v2` embedding model.

### 2. Provider selection

The sidebar offers:

- `None (Offline)` - no external LLM API; answers are extracted locally.
- `Gemini` - uses `gemini-2.5-flash`.
- `OpenAI` - uses `gpt-4o-mini`.

`src/llm.py` creates the selected provider client. If the provider is
`None (Offline)`, or an API key is missing, `get_llm()` returns `None`.
Gemini and OpenAI keys can be entered in the sidebar or supplied through
`GEMINI_API_KEY` and `OPENAI_API_KEY` environment variables.

### 3. Processing uploaded documents

When **Process Documents** is clicked:

1. `src/loader.py` reads each PDF, TXT, or DOCX file.
2. PDF text is extracted page by page; DOCX paragraphs and table rows are
   extracted; TXT files are decoded as UTF-8.
3. Extracted text is cleaned and converted into LangChain `Document` objects
   with source-file and page metadata.
4. `RAGEngine.index()` splits the documents into overlapping chunks.
5. A FAISS vector index is built using MiniLM embeddings.
6. A TF-IDF index is built for keyword matching.
7. The engine, file names, and chunk count are saved in session state.
8. The previous chat and summary are cleared because the document collection
   has changed.

The button reports the number of indexed chunks. If no text can be extracted,
the app displays a warning. Scanned/image-only PDFs require OCR, which this
project does not currently provide.

### 4. Asking a question

After documents have been processed, a question submitted through the chat
box follows this flow:

1. `src/nlp_tools.py` classifies the question intent using regular-expression
   patterns, such as `Summarisation`, `Definition`, `Comparison`, or
   `How-to`.
2. The same module runs a local DistilBERT sentiment model and displays the
   sentiment label and score.
3. Summarisation-intent questions summarize the full indexed document.
4. Other questions call `RAGEngine.answer()`.
5. `RAGEngine.retrieve()` searches a candidate pool with FAISS semantic
   similarity and TF-IDF keyword similarity.
6. Retrieval combines the scores with a 70% semantic and 30% keyword weight,
   then returns the requested `top_k` chunks.
7. Source previews, file names, page numbers, and scores are shown in the
   **Sources** expander.

#### Offline question answering

With `None (Offline)`, no generative LLM is used. The retrieved chunks are
split into sentences, TF-IDF similarity is calculated between each sentence
and the question, and up to three of the most relevant existing sentences
are returned. This is extractive answering: the application selects text from
the documents rather than writing new text.

#### Gemini/OpenAI question answering

With Gemini or OpenAI selected, the retrieved chunks and the recent chat
history are placed into a prompt. The selected LLM generates an answer using
that context and is instructed to cite sources using bracket numbers. If an
LLM request fails, `RAGEngine.answer()` falls back to the local extractive
answer and displays an API-error notice.

### 5. Generating a summary

When **Generate Summary** is clicked after documents have been processed:

1. The full indexed text is obtained from `RAGEngine.full_text()`.
2. The selected sentence count, from 3 to 10, is passed to `summarise()`.
3. In offline mode, TF-IDF sentence scoring selects important sentences.
4. With Gemini or OpenAI, the text is summarized in sections and the section
   summaries are combined into a final summary.
5. The result is stored in session state and displayed in the
   **Document Summary** expander.

The summary button does nothing until **Process Documents** has created an
engine. If an LLM summary fails, the summarization code falls back to an
extractive local summary.

### 6. Chat and session controls

- **Clear Chat** removes the current conversation but keeps the processed
  document index available for new questions.
- Processing documents again rebuilds the index and clears both chat and
  summary state.
- The cached RAG engine is reused, while its indexed contents are replaced
  when a new document set is processed.

## Data and Retrieval Pipeline

The runtime pipeline can be summarized as:

```text
Uploaded files
    |
    v
load_file() -> cleaned Document objects
    |
    v
RecursiveCharacterTextSplitter -> overlapping chunks
    |
    +--> MiniLM embeddings -> FAISS semantic index
    |
    +--> TF-IDF vectorizer -> keyword index
    |
    v
User question
    |
    +--> intent() and sentiment()
    |
    +--> Summarisation intent -> full-document summary
    |
    +--> Other intent -> hybrid retrieval
                              |
                              v
                    offline extraction or LLM answer
                              |
                              v
                    answer plus source previews
```

## Project Structure

- `app.py` - Streamlit web interface
- `src/loader.py` - Text extraction for PDF, TXT, and DOCX files
- `src/rag.py` - Document chunking, FAISS indexing, and hybrid search
- `src/nlp_tools.py` - Extractive summarization, sentiment, and intent classification
- `src/llm.py` - Wrapper for Gemini and OpenAI API calls
- `eval/evaluate.py` - Script to test retrieval accuracy (Hit@K and MRR)

## Running Evaluation

To test retrieval metrics on the sample document:

```bash
python eval/evaluate.py sample_docs/ai_overview.txt eval/eval_set.json
```

The evaluation script indexes the selected document, runs each question from
the JSON file, checks whether expected keywords appear in the retrieved
chunks, and prints:

- `Hit@K`: the fraction of questions whose expected content appears within the
  first `K` results.
- `MRR` (Mean Reciprocal Rank): the average reciprocal rank of the first
  matching result.
