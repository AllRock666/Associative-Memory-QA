import re

def update_app():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()

    sidebar_toggle = """
    st.markdown("### ⚖️ Legal Mode (Experimental)")
    legal_mode = st.toggle("Enable Legal QA Mode", value=False, help="Loads a canned bi-temporal legal corpus to demonstrate the LegalHybridRetriever.")

    st.markdown("### 🧠 Model Settings")"""
    content = content.replace('st.markdown("### 🧠 Model Settings")', sidebar_toggle)

    canned_legal_code = """
def load_canned_legal_pipeline():
    from core.pipeline import HopfieldQAPipeline, PipelineConfig
    from core.embeddings import EmbeddingEngine
    from core.hopfield_torch import PyTorchHopfieldMemory
    from core.bitemporal_graph import BiTemporalGraph
    from core.hybrid_retriever import LegalHybridRetriever
    import datetime
    import numpy as np
    import streamlit as st
    
    if "legal_pipeline" in st.session_state and st.session_state.legal_pipeline is not None:
        return st.session_state.legal_pipeline
        
    config = PipelineConfig(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=512,
        chunk_overlap=64,
        hopfield_beta=8.0,
        top_k=2,
        use_fusion=True
    )
    pipeline = HopfieldQAPipeline(config)
    
    embedder = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    hopfield = PyTorchHopfieldMemory(beta=8.0, n_iter=3)
    graph = BiTemporalGraph()
    
    doc1_text = "The penalty for late delivery is $50,000."
    doc2_text = "Amendment: The penalty for late delivery is increased to $75,000."
    doc3_text = "The governing law for this contract is the state of California."
    
    emb1 = embedder.embed_query(doc1_text)
    emb2 = embedder.embed_query(doc2_text)
    emb3 = embedder.embed_query(doc3_text)
    
    node1_id = graph.add_node(doc1_text, emb1, valid_from=datetime.date(2023, 1, 1))
    node2_id = graph.amend_node(node1_id, doc2_text, emb2, valid_from=datetime.date(2024, 1, 1))
    node3_id = graph.add_node(doc3_text, emb3, valid_from=datetime.date(2023, 1, 1))
    
    patterns = np.array([emb1, emb2, emb3])
    metadata = [
        {"chunk_id": "doc1", "text": doc1_text},
        {"chunk_id": "doc2", "text": doc2_text},
        {"chunk_id": "doc3", "text": doc3_text}
    ]
    hopfield.store(patterns, metadata)
    
    hybrid = LegalHybridRetriever(hopfield, graph, embedder)
    
    st.session_state.legal_pipeline = pipeline
    st.session_state.legal_hybrid = hybrid
    st.session_state.legal_embedder = embedder
    return pipeline

def get_pipeline("""
    content = content.replace("def get_pipeline(", canned_legal_code)

    query_exec = """        if not legal_mode:
            with st.spinner("Retrieving facts and synthesizing answer..."):
                start_t = time.time()
                result = pipeline.generate_answer(query)
                elapsed = (time.time() - start_t) * 1000
        else:
            with st.spinner("Routing query through Bi-Temporal Graph..."):
                start_t = time.time()
                hybrid = st.session_state.legal_hybrid
                ans_text, path = hybrid.route_and_retrieve(query, date_context=None)
                
                from core.pipeline import QAResult
                from core.hopfield_torch import RetrievalResult
                
                fake_chunks = [RetrievalResult(chunk_id=c['id'], text=c['text'], source=c['type'], chunk_index=0, similarity_score=1.0, rank=i, metadata={'source': c['type'], 'text': c['text']}, hopfield_attention=1.0) for i, c in enumerate(path)]
                
                from core.answer_generator import Answer
                answer_obj = Answer(answer=ans_text, confidence=1.0, highlighted_passages=[], context_used=" ".join(c['text'] for c in path))
                
                result = QAResult(
                    query=query,
                    processed_query=None,
                    retrieved_chunks=fake_chunks,
                    answer=answer_obj
                )
                elapsed = (time.time() - start_t) * 1000"""

    content = content.replace("""        with st.spinner("Retrieving facts and synthesizing answer..."):
            start_t = time.time()
            result = pipeline.generate_answer(query)
            elapsed = (time.time() - start_t) * 1000""", query_exec)

    main_logic = """    if legal_mode:
        pipeline = load_canned_legal_pipeline()
        st.session_state.docs_loaded = True
    else:
        pipeline = get_pipeline(
            embedding_model, chunk_size, chunk_overlap, hopfield_beta, top_k
        )

    st.session_state.pipeline = pipeline"""
    content = content.replace("""    pipeline = get_pipeline(
        embedding_model, chunk_size, chunk_overlap, hopfield_beta, top_k
    )
    st.session_state.pipeline = pipeline""", main_logic)

    upload_disable = """    if legal_mode:
        st.info("⚖️ **Legal Mode Active**: A canned bi-temporal dataset of contract amendments is currently loaded. To upload your own documents, disable Legal Mode in the sidebar.")
    else:
        st.markdown("#### 📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs or TXTs",
            type=["txt", "pdf"],
            accept_multiple_files=True
        )"""
    content = content.replace("""    st.markdown("#### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or TXTs",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )""", upload_disable)

    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    update_app()
