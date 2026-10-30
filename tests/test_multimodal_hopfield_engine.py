"""
Test Suite for Multimodal Hopfield Knowledge Intelligence & Visual Learning Engine
===================================================================================
Covers:
1. Multimodal Document Reader (TXT, CSV, XLSX, PPTX, DOCX, PDF, Images)
2. Vision Provider Interface and Graceful Local Fallback
3. AI Visual Studio Generation & Strict Data Integrity
4. Hierarchical Knowledge Memory 2.0 (5 Levels) & Attractor Dynamics
5. Knowledge Graph Multi-Hop Traversal & Cytoscape Serialization
6. AI Study Assistant (Flashcards, MCQs, Rationales, Roadmaps)
7. FastAPI Multimodal Endpoints
"""

import io
import csv
import json
import pytest
import numpy as np
from fastapi.testclient import TestClient

from core.vision_provider import (
    BaseVisionProvider,
    LocalFallbackVisionProvider,
    OpenAIVisionProvider,
    GeminiVisionProvider,
    get_vision_provider,
    VisionAnalysisResult
)
from core.multimodal_reader import (
    MultimodalDocumentReader,
    MultimodalDocument,
    DocumentPage,
    ExtractedElement
)
from core.visual_studio import VisualStudioEngine, VisualizationResult
from core.knowledge_graph import KnowledgeGraphEngine, GraphEntity, GraphRelation
from core.study_assistant import StudyAssistantEngine
from core.hopfield_hierarchical import HierarchicalHopfieldMemory
import api


# ------------------------------------------------------------------------------
# 1. Vision Provider Tests
# ------------------------------------------------------------------------------

def test_local_fallback_vision_provider_does_not_crash():
    provider = LocalFallbackVisionProvider()
    dummy_bytes = b"fake-image-bytes"
    result = provider.analyze_image(dummy_bytes)
    assert result.success is True
    assert "visual" in result.summary.lower()
    assert provider.extract_tables(dummy_bytes) == []
    diag = provider.understand_diagram(dummy_bytes)
    assert diag["type"] == "diagram"


def test_get_vision_provider_factory():
    p = get_vision_provider("local")
    assert isinstance(p, (LocalFallbackVisionProvider, BaseVisionProvider))


# ------------------------------------------------------------------------------
# 2. Multimodal Document Reader Tests
# ------------------------------------------------------------------------------

def test_parse_text_and_markdown():
    reader = MultimodalDocumentReader()
    md_content = b"# Operating Systems\n\nOperating systems coordinate hardware resources.\n\n## Scheduling\n\nCPU scheduling ensures fair allocation."
    doc = reader.parse_file(md_content, "os.md")
    assert doc.total_pages == 1
    assert doc.file_type == "markdown"
    assert len(doc.pages[0].elements) >= 3
    chunks = doc.to_chunks()
    assert len(chunks) >= 1
    assert chunks[0]["page_number"] == 1


def test_parse_csv_tables():
    reader = MultimodalDocumentReader()
    csv_data = b"Framework,Latency,Throughput\nSpark,12.5ms,High\nFlink,4.2ms,Ultra-High\n"
    doc = reader.parse_file(csv_data, "benchmarks.csv")
    assert doc.total_pages == 1
    assert len(doc.tables) == 1
    tbl = doc.tables[0]
    assert tbl["headers"] == ["Framework", "Latency", "Throughput"]
    assert len(tbl["rows"]) == 2
    chunks = doc.to_chunks()
    assert any(c["element_type"] == "table" for c in chunks)


def test_parse_xlsx_sheets():
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Performance"
    ws.append(["Model", "Accuracy", "F1"])
    ws.append(["Hopfield-v2", "98.5%", "0.98"])
    out = io.BytesIO()
    wb.save(out)
    excel_bytes = out.getvalue()

    reader = MultimodalDocumentReader()
    doc = reader.parse_file(excel_bytes, "eval.xlsx")
    assert doc.total_pages == 1
    assert len(doc.tables) == 1
    assert doc.tables[0]["sheet_name"] == "Performance"
    assert doc.tables[0]["headers"] == ["Model", "Accuracy", "F1"]


def test_parse_pptx_slides():
    from pptx import Presentation
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Hopfield Associative Intelligence"
    slide.shapes.placeholders[1].text = "Continuous attractor memory dynamics."
    out = io.BytesIO()
    prs.save(out)
    pptx_bytes = out.getvalue()

    reader = MultimodalDocumentReader()
    doc = reader.parse_file(pptx_bytes, "presentation.pptx")
    assert doc.total_pages == 1
    assert "Hopfield" in doc.full_text


def test_parse_docx_document():
    import docx
    d = docx.Document()
    d.add_heading("Cloud Systems", level=1)
    d.add_paragraph("Kubernetes automates container orchestration.")
    out = io.BytesIO()
    d.save(out)
    docx_bytes = out.getvalue()

    reader = MultimodalDocumentReader()
    doc = reader.parse_file(docx_bytes, "cloud.docx")
    assert doc.total_pages == 1
    assert "Kubernetes" in doc.full_text


# ------------------------------------------------------------------------------
# 3. AI Visual Studio & Data Integrity Tests
# ------------------------------------------------------------------------------

def test_visual_studio_generates_valid_mermaid():
    studio = VisualStudioEngine()
    chunks = [
        {"text": "YARN separates resource management from execution scheduling.", "source": "yarn.txt", "page_number": 1},
        {"text": "MapReduce runs batch workloads while Spark runs memory analytics.", "source": "yarn.txt", "page_number": 2}
    ]

    # Mind map
    res_mm = studio.generate_visualization("Generate a mind map of YARN", chunks)
    assert res_mm.viz_type == "mermaid"
    assert res_mm.subtype == "mindmap"
    assert "mindmap" in res_mm.content

    # Flowchart
    res_fc = studio.generate_visualization("Create a flowchart of processing", chunks)
    assert res_fc.viz_type == "mermaid"
    assert res_fc.subtype == "flowchart"
    assert "flowchart" in res_fc.content

    # Architecture
    res_arch = studio.generate_visualization("Create an architecture diagram", chunks)
    assert "flowchart" in res_arch.content


def test_visual_studio_rejects_hallucinated_charts_honestly():
    studio = VisualStudioEngine()
    # Context with NO numbers
    chunks = [{"text": "Operating systems manage memory, processes, and security.", "source": "os.txt", "page_number": 1}]
    res = studio.generate_visualization("Generate a bar chart comparing performance", chunks)

    # Must NOT invent numbers!
    assert res.has_numerical_data is False
    assert res.suggested_alternative == "mindmap"
    assert "could not be generated" in res.description


def test_visual_studio_generates_plotly_for_verified_numbers():
    studio = VisualStudioEngine()
    chunks = [{"text": "MapReduce: 45.2\nSpark: 12.1\nFlink: 8.4", "source": "bench.txt", "page_number": 1}]
    res = studio.generate_visualization("Generate a bar chart comparing performance", chunks)

    assert res.has_numerical_data is True
    assert res.viz_type == "plotly"
    assert "data" in res.content
    assert len(res.content["data"][0]["x"]) >= 2


# ------------------------------------------------------------------------------
# 4. Knowledge Graph Tests
# ------------------------------------------------------------------------------

def test_knowledge_graph_ingest_and_expand():
    kg = KnowledgeGraphEngine()
    chunks = [
        {"text": "Yet Another Resource Negotiator (YARN) coordinates Hadoop clusters.", "source": "yarn.txt", "doc_id": "d1", "page_number": 1},
        {"text": "Apache Spark uses Resilient Distributed Datasets (RDD).", "source": "spark.txt", "doc_id": "d2", "page_number": 1}
    ]
    kg.extract_and_ingest(chunks)
    assert kg.graph.number_of_nodes() >= 3

    # Multi-hop expansion
    expanded = kg.expand_neighborhood(["YARN", "Hadoop"], max_hops=2)
    assert len(expanded) >= 1

    # Cytoscape export
    cy = kg.to_cytoscape()
    assert "elements" in cy
    assert len(cy["elements"]["nodes"]) >= 3


# ------------------------------------------------------------------------------
# 5. Study Assistant Tests
# ------------------------------------------------------------------------------

def test_study_assistant_generates_deck():
    study = StudyAssistantEngine()
    chunks = [
        {"text": "Continuous Hopfield networks minimize the modern energy function using exponential attention attractors.", "source": "hopfield.pdf", "page_number": 3},
        {"text": "YARN functions as cluster operating system separating resource management from job scheduling.", "source": "yarn.pdf", "page_number": 1}
    ]
    deck = study.generate_study_deck("Distributed Systems", chunks)
    assert deck.deck_id.startswith("deck_")
    assert len(deck.flashcards) >= 1
    assert len(deck.questions) >= 1
    # Check pedagogical distractor rationale
    assert len(deck.questions[0].distractor_rationales) >= 1
    deck_dict = deck.to_dict()
    assert "flashcards" in deck_dict
    assert "questions" in deck_dict


# ------------------------------------------------------------------------------
# 6. Hierarchical Hopfield Memory 2.0 Tests
# ------------------------------------------------------------------------------

def test_hierarchical_hopfield_memory_5_levels():
    memory = HierarchicalHopfieldMemory(embedding_dim=4, beta=8.0)
    fake_chunks = [
        {"chunk_id": "c1", "text": "Continuous Hopfield network memory", "source": "doc1.txt", "page_number": 1},
        {"chunk_id": "c2", "text": "Apache Spark distributed memory analytics", "source": "doc2.txt", "page_number": 2}
    ]
    fake_embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ], dtype=np.float32)

    ingested = memory.ingest_chunks(fake_chunks, fake_embeddings)
    assert ingested == 2

    # Duplicate ingestion must be skipped via content-hash
    ingested_dupes = memory.ingest_chunks(fake_chunks, fake_embeddings)
    assert ingested_dupes == 0

    # Retrieve
    q_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    res = memory.retrieve("Continuous Hopfield", q_emb, top_k=2)
    assert len(res.core_attractors) >= 1
    assert res.core_attractors[0]["chunk_id"] == "c1"
    assert res.is_grounded is True

    # Episodic recording
    memory.record_interaction(
        session_id="s1",
        query="Continuous Hopfield",
        retrieved_memory_ids=["c1"],
        answer_summary="Hopfield continuous network minimizes energy.",
        user_feedback="helpful"
    )
    stats = memory.get_stats()
    assert stats["level1_raw_memory_count"] == 2
    assert stats["level5_episodic_interactions"] == 1


# ------------------------------------------------------------------------------
# 7. FastAPI Multimodal Endpoints Tests
# ------------------------------------------------------------------------------

def test_api_multimodal_document_and_studio_endpoints():
    api.startup_event()
    client = TestClient(api.app)

    # 1. Upload CSV document
    csv_content = b"Algorithm,Complexity,Memory\nQuicksort,O(n log n),O(log n)\nMergesort,O(n log n),O(n)\n"
    res = client.post(
        "/api/upload",
        files={"file": ("algorithms.csv", csv_content, "text/csv")}
    )
    assert res.status_code == 200
    data = res.json()
    assert "doc_id" in data
    doc_id = data["doc_id"]

    # 2. List multimodal documents
    res_m = client.get("/api/documents/multimodal")
    assert res_m.status_code == 200
    docs = res_m.json()["documents"]
    assert any(d["doc_id"] == doc_id for d in docs)

    # 3. Get document pages
    res_p = client.get(f"/api/documents/{doc_id}/pages")
    assert res_p.status_code == 200
    assert len(res_p.json()["pages"]) >= 1

    # 4. Get page details
    res_p1 = client.get(f"/api/documents/{doc_id}/pages/1")
    assert res_p1.status_code == 200
    assert res_p1.json()["has_tables"] is True

    # 5. Table CSV export
    res_tbl = client.get(f"/api/documents/{doc_id}/tables")
    assert res_tbl.status_code == 200
    tables = res_tbl.json()["tables"]
    assert len(tables) >= 1
    tbl_id = tables[0]["table_id"]
    res_csv = client.get(f"/api/documents/{doc_id}/tables/{tbl_id}/csv")
    assert res_csv.status_code == 200
    assert "Quicksort" in res_csv.text

    # 6. Visual Studio endpoint
    res_studio = client.post(
        "/api/visual-studio/generate",
        json={"prompt": "Generate a mind map of algorithms", "requested_type": "mindmap"}
    )
    assert res_studio.status_code == 200
    assert res_studio.json()["viz_type"] == "mermaid"

    # 7. Knowledge graph endpoint
    res_kg = client.get("/api/knowledge-graph")
    assert res_kg.status_code == 200
    assert "elements" in res_kg.json()

    # 8. Study deck endpoint
    res_study = client.post(
        "/api/study/deck",
        json={"topic": "Sorting Algorithms"}
    )
    assert res_study.status_code == 200
    assert "flashcards" in res_study.json()

    # 9. Memory stats endpoint
    res_stats = client.get("/api/memory/hierarchical/stats")
    assert res_stats.status_code == 200
    assert "level1_raw_memory_count" in res_stats.json()
