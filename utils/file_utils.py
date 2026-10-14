"""
File Utilities
==============
Load and extract text from various document formats.
"""

import re
import io
from typing import Tuple, Optional
from pathlib import Path


def extract_text_from_file(file_obj, filename: str) -> Tuple[str, str]:
    """
    Extract text from uploaded file object.

    Returns:
        (text, source_name) tuple
    """
    ext = Path(filename).suffix.lower()
    source = Path(filename).stem

    if ext == ".txt":
        text = file_obj.read().decode("utf-8", errors="replace")
        return _clean_text(text), source

    elif ext == ".pdf":
        return _extract_pdf(file_obj), source

    elif ext == ".docx":
        return _extract_docx(file_obj), source

    elif ext in (".md", ".markdown"):
        text = file_obj.read().decode("utf-8", errors="replace")
        # Strip markdown formatting
        text = re.sub(r'#{1,6}\s+', '', text)
        text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)
        return _clean_text(text), source

    else:
        # Try to read as plain text
        try:
            text = file_obj.read().decode("utf-8", errors="replace")
            return _clean_text(text), source
        except Exception:
            raise ValueError(f"Unsupported file format: {ext}")


def _clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Normalize unicode
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Remove multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove non-printable characters
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]', ' ', text)
    # Collapse multiple spaces (but not newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def _extract_pdf(file_obj) -> str:
    """
    Extract text from PDF using pypdf, PyPDF2, or other available readers.
    Supports decryption of blank passwords and tolerates malformed pages.
    """
    pdf_module = None
    try:
        import pypdf
        pdf_module = pypdf
    except ImportError:
        try:
            import PyPDF2
            pdf_module = PyPDF2
        except ImportError:
            pass

    last_error = None
    if pdf_module is not None:
        try:
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            try:
                reader = pdf_module.PdfReader(file_obj, strict=False)
            except TypeError:
                reader = pdf_module.PdfReader(file_obj)
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")
                except Exception:
                    pass
            pages = []
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        pages.append(page_text)
                except Exception as pe:
                    continue
            if pages:
                return _clean_text('\n\n'.join(pages))
        except Exception as e:
            last_error = e

    # Fallback to PyMuPDF (fitz) if installed
    try:
        import fitz
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        data = file_obj.read() if hasattr(file_obj, "read") else file_obj
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for page in doc:
            t = page.get_text()
            if t and t.strip():
                pages.append(t)
        if pages:
            return _clean_text('\n\n'.join(pages))
    except Exception:
        pass

    # Fallback to pdfplumber if installed
    try:
        import pdfplumber
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        with pdfplumber.open(file_obj) as pdf:
            pages = []
            for p in pdf.pages:
                t = p.extract_text()
                if t and t.strip():
                    pages.append(t)
            if pages:
                return _clean_text('\n\n'.join(pages))
    except Exception:
        pass

    if pdf_module is None:
        raise ImportError("No PDF parser installed. Please install pypdf: pip install pypdf")

    err_detail = f" (Details: {last_error})" if last_error else ""
    raise ValueError(
        f"No extractable text found in PDF{err_detail}. The document may be empty, password-protected, or a scanned image requiring OCR."
    )


def _extract_docx(file_obj) -> str:
    """Extract text from DOCX using python-docx."""
    try:
        from docx import Document
        doc = Document(file_obj)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return _clean_text('\n\n'.join(paragraphs))
    except ImportError:
        raise ImportError("python-docx required for DOCX support: pip install python-docx")
    except Exception as e:
        raise ValueError(f"Failed to extract DOCX text: {e}")


def estimate_chunk_count(text: str, chunk_size: int = 512, overlap: int = 64) -> int:
    """Estimate number of chunks that will be generated."""
    if not text:
        return 0
    effective_chunk = chunk_size - overlap
    return max(1, len(text) // effective_chunk)
