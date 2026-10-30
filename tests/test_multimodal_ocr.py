import io
import pytest
from PIL import Image, ImageDraw
from core.vision_provider import LocalOCREngine, LocalFallbackVisionProvider, get_vision_provider
from core.multimodal_reader import MultimodalDocumentReader, MultimodalDocument


def make_test_image(text: str = "Hopfield Network Architecture") -> bytes:
    """Creates a high-contrast synthetic image with clear text for testing OCR."""
    img = Image.new("RGB", (600, 150), color="white")
    draw = ImageDraw.Draw(img)
    # Draw text cleanly with high contrast
    draw.text((30, 50), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_ocr_engine_blank_image():
    """A completely blank image should return empty text without error."""
    blank = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    blank.save(buf, format="PNG")
    res = LocalOCREngine.extract_text(buf.getvalue())
    assert isinstance(res, dict)
    assert "text" in res
    assert "lines" in res


def test_local_fallback_vision_provider_structure():
    """LocalFallbackVisionProvider should return a valid VisionAnalysisResult."""
    img_bytes = make_test_image("Associative Memory 2026")
    provider = LocalFallbackVisionProvider()
    result = provider.analyze_image(img_bytes)
    assert result.success is True
    assert isinstance(result.key_concepts, list)
    assert "image" in result.summary.lower() or "processed" in result.summary.lower()


def test_multimodal_reader_parses_image():
    """MultimodalDocumentReader should parse images and generate text elements & chunks."""
    img_bytes = make_test_image("Deep Learning Multimodal Index")
    reader = MultimodalDocumentReader()
    doc = reader.parse_file(img_bytes, "test_diagram.png")

    assert doc is not None
    assert doc.file_type == "image"
    assert doc.total_pages == 1
    assert len(doc.pages) == 1
    page = doc.pages[0]
    assert page.has_images is True

    # Chunks must be generated for indexing
    chunks = doc.to_chunks()
    assert len(chunks) > 0
    assert chunks[0]["source"] == "test_diagram.png"
    assert "doc_id" in chunks[0]
