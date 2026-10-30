"""Step 1 of RAG: read your documents and return their text page by page."""
from pathlib import Path


def load_documents(folder):
    """Return a list of pages: {"text", "source", "page"} for every PDF/TXT/MD file."""
    folder = Path(folder)
    pages = []
    for path in sorted(folder.glob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages.extend(_read_pdf(path))
        elif suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                pages.append({"text": text, "source": path.name, "page": 1})
    return pages


def _read_pdf(path):
    from pypdf import PdfReader  # imported here so the rest works without pypdf

    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "source": path.name, "page": number})
    return pages
