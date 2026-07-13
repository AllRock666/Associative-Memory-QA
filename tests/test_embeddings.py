import pytest
from core.embeddings import DocumentChunker

def test_chunker_basic():
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=10)
    text = "This is a short text. It will be chunked into pieces."
    chunks = chunker.chunk_text(text, source="test")
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.text) >= 10
        assert chunk.source == "test"
        
def test_chunker_empty():
    chunker = DocumentChunker()
    chunks = chunker.chunk_text("")
    assert len(chunks) == 0
