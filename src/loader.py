import io
import re
from langchain_core.documents import Document

SUPPORTED_TYPES = ("pdf", "txt", "docx")


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_file(filename, file_bytes):
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file format: .{ext}")

    docs = []
    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        for page_num, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                docs.append(Document(page_content=text, metadata={"source": filename, "page": page_num}))

    elif ext == "docx":
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        lines = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                lines.append(" | ".join(cell.text.strip() for cell in row.cells))
        text = clean_text("\n".join(lines))
        if text:
            docs.append(Document(page_content=text, metadata={"source": filename, "page": 1}))

    else:
        text = clean_text(file_bytes.decode("utf-8", errors="replace"))
        if text:
            docs.append(Document(page_content=text, metadata={"source": filename, "page": 1}))

    return docs
