"""
Multimodal Document Reader Engine
=================================
Comprehensive hybrid extraction pipeline supporting:
- PDF (PyMuPDF for layout/text/images + pdfplumber for accurate table extraction)
- DOCX (python-docx paragraphs and tables)
- PPTX (python-pptx slides, text shapes, tables, notes)
- XLSX & CSV (openpyxl / csv tabular extraction)
- Plain Text & Markdown (.txt, .md)
- Images (.png, .jpg, .jpeg, .webp)

Preserves exact page numbers, element types, coordinates, and provenance.
Seamlessly integrates with VisionProvider for page summaries and diagram descriptions.
"""

import io
import os
import re
import csv
import json
import uuid
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

from core.vision_provider import BaseVisionProvider, get_vision_provider, VisionAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class ExtractedElement:
    """An individual extracted component from a document."""
    element_id: str
    element_type: str  # "text" | "heading" | "table" | "diagram" | "image" | "sheet"
    content: str
    page_number: int
    doc_id: str
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    metadata: Dict[str, Any] = field(default_factory=dict)
    structured_data: Optional[Any] = None  # e.g., table dict {headers: [], rows: []}


@dataclass
class DocumentPage:
    """Representation of an individual document page or slide."""
    page_number: int
    text_content: str
    elements: List[ExtractedElement] = field(default_factory=list)
    has_tables: bool = False
    has_images: bool = False
    page_image_bytes: Optional[bytes] = None  # Rendered image for PDF/slides
    ai_summary: Optional[str] = None
    key_concepts: List[str] = field(default_factory=list)


@dataclass
class MultimodalDocument:
    """Complete parsed multimodal document with rich provenance."""
    doc_id: str
    filename: str
    file_type: str
    total_pages: int
    pages: List[DocumentPage] = field(default_factory=list)
    full_text: str = ""
    tables: List[Dict[str, Any]] = field(default_factory=list)
    diagrams: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_page(self, page_number: int) -> Optional[DocumentPage]:
        for p in self.pages:
            if p.page_number == page_number:
                return p
        return None

    def to_chunks(self, max_chunk_size: int = 512, overlap: int = 64) -> List[Dict[str, Any]]:
        """
        Converts the document's elements into enriched chunks with full provenance metadata
        (doc_id, page_number, element_type, source_table, etc.).
        """
        chunks = []
        for page in self.pages:
            # 1. Chunk tables as distinct self-contained units
            for elem in page.elements:
                if elem.element_type == "table":
                    tbl_text = elem.content
                    chunks.append({
                        "text": tbl_text,
                        "source": self.filename,
                        "doc_id": self.doc_id,
                        "page_number": page.page_number,
                        "element_type": "table",
                        "metadata": {
                            "table_id": elem.element_id,
                            "structured_table": elem.structured_data,
                            "page": page.page_number
                        }
                    })

            # 2. Chunk text flow
            text_elems = [e for e in page.elements if e.element_type in ("text", "heading") and e.content and e.content.strip()]
            if not text_elems:
                text_elems = [e for e in page.elements if e.element_type == "image" and e.content and e.content.strip()]
            if text_elems:
                combined_page_text = "\n\n".join(e.content for e in text_elems if e.content.strip())
                if len(combined_page_text) <= max_chunk_size:
                    chunks.append({
                        "text": combined_page_text,
                        "source": self.filename,
                        "doc_id": self.doc_id,
                        "page_number": page.page_number,
                        "element_type": "text",
                        "metadata": {"page": page.page_number}
                    })
                else:
                    # Token-like or word-like sliding window
                    words = combined_page_text.split()
                    step = max_chunk_size - overlap
                    for i in range(0, len(words), max(1, step)):
                        chunk_words = words[i:i + max_chunk_size]
                        if not chunk_words:
                            continue
                        chunk_str = " ".join(chunk_words)
                        chunks.append({
                            "text": chunk_str,
                            "source": self.filename,
                            "doc_id": self.doc_id,
                            "page_number": page.page_number,
                            "element_type": "text",
                            "metadata": {"page": page.page_number, "offset": i}
                        })
        return chunks


class MultimodalDocumentReader:
    """Hybrid document intelligence parser."""

    def __init__(self, vision_provider: Optional[BaseVisionProvider] = None):
        self.vision_provider = vision_provider or get_vision_provider()

    def parse_file(self, file_content: Union[bytes, io.BytesIO], filename: str) -> MultimodalDocument:
        """Parses any supported file type and extracts multimodal elements."""
        if isinstance(file_content, io.BytesIO):
            file_bytes = file_content.getvalue()
        elif isinstance(file_content, bytes):
            file_bytes = file_content
        else:
            file_bytes = file_content.read()

        ext = Path(filename).suffix.lower()
        doc_id = hashlib.sha256(file_bytes).hexdigest()[:16]

        if ext == ".pdf":
            return self._parse_pdf(file_bytes, filename, doc_id)
        elif ext in (".docx", ".doc"):
            return self._parse_docx(file_bytes, filename, doc_id)
        elif ext in (".pptx", ".ppt"):
            return self._parse_pptx(file_bytes, filename, doc_id)
        elif ext in (".xlsx", ".xls"):
            return self._parse_xlsx(file_bytes, filename, doc_id)
        elif ext == ".csv":
            return self._parse_csv(file_bytes, filename, doc_id)
        elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            return self._parse_image(file_bytes, filename, doc_id)
        elif ext in (".txt", ".md", ".markdown"):
            return self._parse_text(file_bytes, filename, doc_id, is_markdown=(ext != ".txt"))
        else:
            # Fallback to plain text decoding
            try:
                return self._parse_text(file_bytes, filename, doc_id, is_markdown=False)
            except Exception:
                raise ValueError(f"Unsupported file format: {ext}")

    def _parse_pdf(self, file_bytes: bytes, filename: str, doc_id: str) -> MultimodalDocument:
        """Hybrid PDF parsing using PyMuPDF and pdfplumber."""
        pages: List[DocumentPage] = []
        all_tables: List[Dict[str, Any]] = []
        all_diagrams: List[Dict[str, Any]] = []

        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            total_pages = len(doc)

            # Plumber fallback for table extraction
            plumber_doc = None
            try:
                import pdfplumber
                plumber_doc = pdfplumber.open(io.BytesIO(file_bytes))
            except Exception as pe:
                logger.debug(f"pdfplumber not available or failed: {pe}")

            for page_idx in range(total_pages):
                page_num = page_idx + 1
                fitz_page = doc[page_idx]
                elements: List[ExtractedElement] = []

                # 1. Native text extraction via PyMuPDF blocks
                blocks = fitz_page.get_text("blocks")
                page_text_parts = []
                for b in blocks:
                    # b: (x0, y0, x1, y1, text, block_no, block_type)
                    b_text = b[4].strip()
                    if not b_text:
                        continue
                    b_type = "heading" if (b[3] - b[1] > 20 and len(b_text) < 100 and not b_text.endswith('.')) else "text"
                    elem_id = f"{doc_id}_p{page_num}_b{b[5]}"
                    elements.append(ExtractedElement(
                        element_id=elem_id,
                        element_type=b_type,
                        content=b_text,
                        page_number=page_num,
                        doc_id=doc_id,
                        bbox=(b[0], b[1], b[2], b[3])
                    ))
                    page_text_parts.append(b_text)

                combined_page_text = "\n\n".join(page_text_parts)

                # 2. Extract tables via pdfplumber
                has_tables = False
                if plumber_doc and page_idx < len(plumber_doc.pages):
                    try:
                        p_page = plumber_doc.pages[page_idx]
                        extracted_tables = p_page.extract_tables()
                        for t_idx, table_data in enumerate(extracted_tables):
                            if not table_data or len(table_data) < 2:
                                continue
                            has_tables = True
                            headers = [str(c or "").strip() for c in table_data[0]]
                            rows = [[str(c or "").strip() for c in row] for row in table_data[1:]]
                            
                            # Format table text
                            tbl_str = f"Table on Page {page_num}:\n"
                            tbl_str += " | ".join(headers) + "\n"
                            tbl_str += " | ".join(["---"] * len(headers)) + "\n"
                            for r in rows:
                                tbl_str += " | ".join(r) + "\n"

                            t_dict = {
                                "table_id": f"{doc_id}_p{page_num}_tbl{t_idx}",
                                "page_number": page_num,
                                "headers": headers,
                                "rows": rows,
                                "text": tbl_str
                            }
                            all_tables.append(t_dict)
                            elements.append(ExtractedElement(
                                element_id=t_dict["table_id"],
                                element_type="table",
                                content=tbl_str,
                                page_number=page_num,
                                doc_id=doc_id,
                                structured_data=t_dict
                            ))
                    except Exception as te:
                        logger.debug(f"Table extraction error on page {page_num}: {te}")

                # 3. Render page image pixmap for inspection and vision
                page_img_bytes = None
                has_images = len(fitz_page.get_images()) > 0
                try:
                    pix = fitz_page.get_pixmap(dpi=150)
                    page_img_bytes = pix.tobytes("png")
                except Exception as pix_err:
                    logger.debug(f"Pixmap render warning on page {page_num}: {pix_err}")

                # 4. Scanned PDF fallback: If virtually no text was extracted, run vision/OCR
                ai_summary = None
                key_concepts = []
                if len(combined_page_text.strip()) < 40 and page_img_bytes:
                    analysis = self.vision_provider.analyze_pdf_page(page_img_bytes, page_number=page_num)
                    if analysis.extracted_text:
                        combined_page_text = analysis.extracted_text
                        elements.append(ExtractedElement(
                            element_id=f"{doc_id}_p{page_num}_ocr",
                            element_type="text",
                            content=analysis.extracted_text,
                            page_number=page_num,
                            doc_id=doc_id
                        ))
                    ai_summary = analysis.summary
                    key_concepts = analysis.key_concepts

                doc_page = DocumentPage(
                    page_number=page_num,
                    text_content=combined_page_text,
                    elements=elements,
                    has_tables=has_tables,
                    has_images=has_images,
                    page_image_bytes=page_img_bytes,
                    ai_summary=ai_summary,
                    key_concepts=key_concepts
                )
                pages.append(doc_page)

            if plumber_doc:
                plumber_doc.close()
            doc.close()

            full_text = "\n\n".join(p.text_content for p in pages if p.text_content)
            return MultimodalDocument(
                doc_id=doc_id,
                filename=filename,
                file_type="pdf",
                total_pages=total_pages,
                pages=pages,
                full_text=full_text,
                tables=all_tables,
                diagrams=all_diagrams,
                metadata={"title": filename, "source": filename, "format": "pdf"}
            )

        except Exception as e:
            logger.warning(f"PyMuPDF failed, falling back to basic extraction: {e}")
            from utils.file_utils import extract_text_from_file
            txt, _ = extract_text_from_file(io.BytesIO(file_bytes), filename)
            elem = ExtractedElement(
                element_id=f"{doc_id}_p1_fallback",
                element_type="text",
                content=txt,
                page_number=1,
                doc_id=doc_id
            )
            page = DocumentPage(page_number=1, text_content=txt, elements=[elem])
            return MultimodalDocument(
                doc_id=doc_id,
                filename=filename,
                file_type="pdf",
                total_pages=1,
                pages=[page],
                full_text=txt,
                metadata={"title": filename, "source": filename}
            )

    def _fallback_multimodal_doc(self, file_bytes: bytes, filename: str, doc_id: str, file_type: str) -> MultimodalDocument:
        from utils.file_utils import extract_text_from_file
        txt, _ = extract_text_from_file(io.BytesIO(file_bytes), filename)
        elem = ExtractedElement(
            element_id=f"{doc_id}_p1_fallback",
            element_type="text",
            content=txt,
            page_number=1,
            doc_id=doc_id
        )
        page = DocumentPage(page_number=1, text_content=txt, elements=[elem])
        return MultimodalDocument(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            total_pages=1,
            pages=[page],
            full_text=txt,
            metadata={"title": filename, "source": filename}
        )

    def _parse_docx(self, file_bytes: bytes, filename: str, doc_id: str) -> MultimodalDocument:
        """Parses DOCX document paragraphs, headings, and tables."""
        try:
            import docx
        except ImportError as e:
            logger.warning(f"python-docx not available, falling back to basic extraction: {e}")
            return self._fallback_multimodal_doc(file_bytes, filename, doc_id, 'docx')
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            elements: List[ExtractedElement] = []
            all_tables: List[Dict[str, Any]] = []
            text_parts = []

            page_num = 1
            for p_idx, para in enumerate(doc.paragraphs):
                txt = para.text.strip()
                if not txt:
                    continue
                is_heading = para.style.name.startswith("Heading") if para.style else False
                elem_type = "heading" if is_heading else "text"
                elem = ExtractedElement(
                    element_id=f"{doc_id}_p1_p{p_idx}",
                    element_type=elem_type,
                    content=txt,
                    page_number=page_num,
                    doc_id=doc_id,
                    metadata={"style": para.style.name if para.style else "Normal"}
                )
                elements.append(elem)
                text_parts.append(txt)

            for t_idx, tbl in enumerate(doc.tables):
                headers = [cell.text.strip() for cell in tbl.rows[0].cells] if tbl.rows else []
                rows = []
                for r in tbl.rows[1:]:
                    rows.append([cell.text.strip() for cell in r.cells])
            
                tbl_str = f"Table {t_idx + 1}:\n" + " | ".join(headers) + "\n" + " | ".join(["---"] * len(headers)) + "\n"
                for r in rows:
                    tbl_str += " | ".join(r) + "\n"

                t_dict = {
                    "table_id": f"{doc_id}_tbl{t_idx}",
                    "page_number": 1,
                    "headers": headers,
                    "rows": rows,
                    "text": tbl_str
                }
                all_tables.append(t_dict)
                elements.append(ExtractedElement(
                    element_id=t_dict["table_id"],
                    element_type="table",
                    content=tbl_str,
                    page_number=1,
                    doc_id=doc_id,
                    structured_data=t_dict
                ))
                text_parts.append(tbl_str)

            combined_text = "\n\n".join(text_parts)
            page = DocumentPage(
                page_number=1,
                text_content=combined_text,
                elements=elements,
                has_tables=len(all_tables) > 0
            )
            return MultimodalDocument(
                doc_id=doc_id,
                filename=filename,
                file_type="docx",
                total_pages=1,
                pages=[page],
                full_text=combined_text,
                tables=all_tables,
                metadata={"title": filename, "source": filename}
            )
        except Exception as e:
            logger.warning(f"DOCX extraction failed, falling back to basic extraction: {e}")
            return self._fallback_multimodal_doc(file_bytes, filename, doc_id, 'docx')

    def _parse_pptx(self, file_bytes: bytes, filename: str, doc_id: str) -> MultimodalDocument:
        """Parses PPTX slides, text shapes, tables, and notes."""
        try:
            from pptx import Presentation
        except ImportError as e:
            logger.warning(f"python-pptx not available, falling back to basic extraction: {e}")
            return self._fallback_multimodal_doc(file_bytes, filename, doc_id, 'pptx')
        try:
            prs = Presentation(io.BytesIO(file_bytes))
            pages: List[DocumentPage] = []
            all_tables: List[Dict[str, Any]] = []

            for slide_idx, slide in enumerate(prs.slides):
                slide_num = slide_idx + 1
                slide_elements: List[ExtractedElement] = []
                slide_texts = []

                for shape_idx, shape in enumerate(slide.shapes):
                    if shape.has_text_frame:
                        for para in shape.text_frame.paragraphs:
                            txt = para.text.strip()
                            if txt:
                                elem = ExtractedElement(
                                    element_id=f"{doc_id}_s{slide_num}_t{shape_idx}",
                                    element_type="text",
                                    content=txt,
                                    page_number=slide_num,
                                    doc_id=doc_id
                                )
                                slide_elements.append(elem)
                                slide_texts.append(txt)

                    if shape.has_table:
                        tbl = shape.table
                        headers = [c.text.strip() for c in tbl.rows[0].cells] if tbl.rows else []
                        rows = [[c.text.strip() for c in r.cells] for r in tbl.rows[1:]]
                        tbl_str = f"Slide {slide_num} Table:\n" + " | ".join(headers) + "\n"
                        for r in rows:
                            tbl_str += " | ".join(r) + "\n"

                        t_dict = {
                            "table_id": f"{doc_id}_s{slide_num}_tbl{shape_idx}",
                            "page_number": slide_num,
                            "headers": headers,
                            "rows": rows,
                            "text": tbl_str
                        }
                        all_tables.append(t_dict)
                        slide_elements.append(ExtractedElement(
                            element_id=t_dict["table_id"],
                            element_type="table",
                            content=tbl_str,
                            page_number=slide_num,
                            doc_id=doc_id,
                            structured_data=t_dict
                        ))
                        slide_texts.append(tbl_str)

                # Check for speaker notes
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes = slide.notes_slide.notes_text_frame.text.strip()
                    if notes:
                        slide_elements.append(ExtractedElement(
                            element_id=f"{doc_id}_s{slide_num}_notes",
                            element_type="text",
                            content=f"Speaker Notes: {notes}",
                            page_number=slide_num,
                            doc_id=doc_id
                        ))
                        slide_texts.append(f"Speaker Notes: {notes}")

                combined_slide_text = "\n\n".join(slide_texts)
                pages.append(DocumentPage(
                    page_number=slide_num,
                    text_content=combined_slide_text,
                    elements=slide_elements,
                    has_tables=len([e for e in slide_elements if e.element_type == "table"]) > 0
                ))

            full_text = "\n\n".join(p.text_content for p in pages if p.text_content)
            return MultimodalDocument(
                doc_id=doc_id,
                filename=filename,
                file_type="pptx",
                total_pages=len(pages),
                pages=pages,
                full_text=full_text,
                tables=all_tables,
                metadata={"title": filename, "source": filename}
            )
        except Exception as e:
            logger.warning(f"PPTX extraction failed, falling back to basic extraction: {e}")
            return self._fallback_multimodal_doc(file_bytes, filename, doc_id, 'pptx')

    def _parse_xlsx(self, file_bytes: bytes, filename: str, doc_id: str) -> MultimodalDocument:
        """Parses Excel spreadsheets sheet-by-sheet into structured tables."""
        try:
            import openpyxl
        except ImportError as e:
            logger.warning(f"openpyxl not available, falling back to basic extraction: {e}")
            return self._fallback_multimodal_doc(file_bytes, filename, doc_id, 'xlsx')
        try:
            wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
            pages: List[DocumentPage] = []
            all_tables: List[Dict[str, Any]] = []

            for sheet_idx, sheet_name in enumerate(wb.sheetnames):
                page_num = sheet_idx + 1
                ws = wb[sheet_name]
                rows_data = []
                for row in ws.iter_rows(values_only=True):
                    if any(v is not None for v in row):
                        rows_data.append([str(v).strip() if v is not None else "" for v in row])

                if not rows_data:
                    continue

                headers = rows_data[0]
                data_rows = rows_data[1:]

                tbl_str = f"Sheet '{sheet_name}':\n"
                tbl_str += " | ".join(headers) + "\n"
                tbl_str += " | ".join(["---"] * len(headers)) + "\n"
                for r in data_rows:
                    tbl_str += " | ".join(r) + "\n"

                t_dict = {
                    "table_id": f"{doc_id}_sheet_{sheet_idx}",
                    "page_number": page_num,
                    "sheet_name": sheet_name,
                    "headers": headers,
                    "rows": data_rows,
                    "text": tbl_str
                }
                all_tables.append(t_dict)

                elem = ExtractedElement(
                    element_id=t_dict["table_id"],
                    element_type="table",
                    content=tbl_str,
                    page_number=page_num,
                    doc_id=doc_id,
                    structured_data=t_dict,
                    metadata={"sheet_name": sheet_name}
                )

                pages.append(DocumentPage(
                    page_number=page_num,
                    text_content=tbl_str,
                    elements=[elem],
                    has_tables=True
                ))

            full_text = "\n\n".join(p.text_content for p in pages)
            return MultimodalDocument(
                doc_id=doc_id,
                filename=filename,
                file_type="xlsx",
                total_pages=len(pages),
                pages=pages,
                full_text=full_text,
                tables=all_tables,
                metadata={"title": filename, "source": filename}
            )
        except Exception as e:
            logger.warning(f"XLSX extraction failed, falling back to basic extraction: {e}")
            return self._fallback_multimodal_doc(file_bytes, filename, doc_id, 'xlsx')

    def _parse_csv(self, file_bytes: bytes, filename: str, doc_id: str) -> MultimodalDocument:
        """Parses CSV tabular data."""
        text = file_bytes.decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)

        if not rows:
            return MultimodalDocument(doc_id=doc_id, filename=filename, file_type="csv", total_pages=0)

        headers = [c.strip() for c in rows[0]]
        data_rows = [[c.strip() for c in r] for r in rows[1:]]

        tbl_str = f"CSV Document: {filename}\n"
        tbl_str += " | ".join(headers) + "\n"
        tbl_str += " | ".join(["---"] * len(headers)) + "\n"
        for r in data_rows:
            tbl_str += " | ".join(r) + "\n"

        t_dict = {
            "table_id": f"{doc_id}_csv",
            "page_number": 1,
            "headers": headers,
            "rows": data_rows,
            "text": tbl_str
        }
        elem = ExtractedElement(
            element_id=t_dict["table_id"],
            element_type="table",
            content=tbl_str,
            page_number=1,
            doc_id=doc_id,
            structured_data=t_dict
        )
        page = DocumentPage(page_number=1, text_content=tbl_str, elements=[elem], has_tables=True)

        return MultimodalDocument(
            doc_id=doc_id,
            filename=filename,
            file_type="csv",
            total_pages=1,
            pages=[page],
            full_text=tbl_str,
            tables=[t_dict],
            metadata={"title": filename, "source": filename}
        )

    def _parse_image(self, file_bytes: bytes, filename: str, doc_id: str) -> MultimodalDocument:
        """Processes an image file via PIL and VisionProvider (with local OCR)."""
        from PIL import Image
        img = Image.open(io.BytesIO(file_bytes))
        w, h = img.size
        img_format = img.format or "PNG"

        analysis = self.vision_provider.analyze_image(file_bytes)
        desc = analysis.summary or f"Image '{filename}' ({w}x{h}, {img_format})"
        
        elements: List[ExtractedElement] = []
        img_elem = ExtractedElement(
            element_id=f"{doc_id}_img",
            element_type="image",
            content=desc,
            page_number=1,
            doc_id=doc_id,
            metadata={"width": w, "height": h, "format": img_format}
        )
        elements.append(img_elem)

        page_text = desc
        if analysis.extracted_text:
            page_text = f"Extracted Text from {filename}:\n\n{analysis.extracted_text}\n\nVisual Summary: {analysis.summary}"
            ocr_elem = ExtractedElement(
                element_id=f"{doc_id}_ocr",
                element_type="text",
                content=analysis.extracted_text,
                page_number=1,
                doc_id=doc_id,
                metadata={"ocr_engine": analysis.provider_used}
            )
            elements.append(ocr_elem)
        else:
            text_elem = ExtractedElement(
                element_id=f"{doc_id}_summary",
                element_type="text",
                content=desc,
                page_number=1,
                doc_id=doc_id
            )
            elements.append(text_elem)

        page = DocumentPage(
            page_number=1,
            text_content=page_text,
            elements=elements,
            has_images=True,
            page_image_bytes=file_bytes,
            ai_summary=analysis.summary,
            key_concepts=analysis.key_concepts
        )
        return MultimodalDocument(
            doc_id=doc_id,
            filename=filename,
            file_type="image",
            total_pages=1,
            pages=[page],
            full_text=page_text,
            tables=analysis.detected_tables,
            diagrams=analysis.detected_diagrams,
            metadata={"title": filename, "source": filename, "width": w, "height": h, "ocr_extracted": bool(analysis.extracted_text)}
        )

    def _parse_text(self, file_bytes: bytes, filename: str, doc_id: str, is_markdown: bool = False) -> MultimodalDocument:
        """Parses plain text and markdown documents."""
        text = file_bytes.decode("utf-8", errors="replace")
        from utils.file_utils import _clean_text
        clean = _clean_text(text)

        elements: List[ExtractedElement] = []
        paragraphs = [p.strip() for p in clean.split("\n\n") if p.strip()]

        for idx, p in enumerate(paragraphs):
            is_head = p.startswith("#") or (len(p) < 80 and not p.endswith("."))
            elem_type = "heading" if is_head else "text"
            clean_p = p.lstrip("#").strip() if is_head else p
            elem = ExtractedElement(
                element_id=f"{doc_id}_p1_para{idx}",
                element_type=elem_type,
                content=clean_p,
                page_number=1,
                doc_id=doc_id
            )
            elements.append(elem)

        page = DocumentPage(
            page_number=1,
            text_content=clean,
            elements=elements
        )
        return MultimodalDocument(
            doc_id=doc_id,
            filename=filename,
            file_type="markdown" if is_markdown else "txt",
            total_pages=1,
            pages=[page],
            full_text=clean,
            metadata={"title": filename, "source": filename}
        )
