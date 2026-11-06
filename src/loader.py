"""Document loading + text cleaning for PDF / TXT / DOCX."""
from __future__ import annotations

import io
import re

from langchain_core.documents import Document

SUPPORTED = ("pdf", "txt", "docx")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)      # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_file(name: str, data: bytes) -> list[Document]:
    """Return one Document per PDF page (or one for TXT/DOCX)."""
    ext = name.lower().rsplit(".", 1)[-1]
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: .{ext}")

    docs: list[Document] = []
    if ext == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        for i, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                docs.append(Document(page_content=text, metadata={"source": name, "page": i}))
    elif ext == "docx":
        import docx

        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs if p.text.strip()]
        for table in d.tables:
            for row in table.rows:
                parts.append(" | ".join(c.text.strip() for c in row.cells))
        text = clean_text("\n".join(parts))
        if text:
            docs.append(Document(page_content=text, metadata={"source": name, "page": 1}))
    else:  # txt
        text = clean_text(data.decode("utf-8", errors="replace"))
        if text:
            docs.append(Document(page_content=text, metadata={"source": name, "page": 1}))
    return docs
