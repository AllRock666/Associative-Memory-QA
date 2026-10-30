import sys
import os
import time
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from core.multimodal_reader import MultimodalDocumentReader

# ---------------------------------------------------------
# PAGE CONFIG & CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hopfield Search - Multimodal Associative Memory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .stChatFloatingInputContainer {
        padding-bottom: 2rem;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
    }
    
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #4B5563;
    }
    
    .doc-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        background-color: #F3F4F6;
        color: #374151;
        font-size: 0.875rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid #E5E7EB;
    }

    .banner-box {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1E40AF;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# STATE INITIALIZATION
# ---------------------------------------------------------
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "docs_loaded" not in st.session_state:
        st.session_state.docs_loaded = False
    if "multimodal_reader" not in st.session_state:
        st.session_state.multimodal_reader = MultimodalDocumentReader()

init_state()

# ---------------------------------------------------------
# PIPELINE INITIALIZATION
# ---------------------------------------------------------
def get_pipeline(embedding_model, chunk_size, chunk_overlap, hopfield_beta, top_k, energy_type, use_projection):
    from core.pipeline import HopfieldQAPipeline, PipelineConfig
    
    if st.session_state.pipeline is None:
        proj_path = "models/projection_squad.pt" if use_projection and os.path.exists("models/projection_squad.pt") else None
        config = PipelineConfig(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            hopfield_beta=hopfield_beta,
            top_k=top_k,
            use_fusion=True,
            use_projection=use_projection,
            projection_path=proj_path,
            llm_provider=os.environ.get("LLM_PROVIDER", "ollama").lower().strip(),
            llm_base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").strip(),
            llm_model=os.environ.get("LLM_MODEL", "llama3.2").strip(),
            llm_api_key=os.environ.get("LLM_API_KEY", "").strip() or None,
        )
        config.energy_type = energy_type
        st.session_state.pipeline = HopfieldQAPipeline(config)
    else:
        p = st.session_state.pipeline
        p.config.hopfield_beta = hopfield_beta
        p.config.top_k = top_k
        p.config.chunk_size = chunk_size
        p.config.energy_type = energy_type
        p.config.use_projection = use_projection
        p.memory.retriever.beta = hopfield_beta
        p.memory.retriever.energy_type = energy_type
        
    return st.session_state.pipeline

# ---------------------------------------------------------
# SIDEBAR (Data & Config)
# ---------------------------------------------------------
with st.sidebar:
    st.title("🧠 Hopfield Search")
    st.markdown("Neural Associative Memory & Multimodal QA")

    st.markdown(
        '<div class="banner-box">✨ <b>Modern Multimodal UI:</b> Run <code>python api.py</code> and open <a href="http://localhost:8000" target="_blank">http://localhost:8000</a> for Visual Studio Diagrams, AI Flashcards & Knowledge Graphs.</div>',
        unsafe_allow_html=True
    )
    st.divider()
    
    st.subheader("1. System Configuration")
    with st.expander("⚙️ Advanced Settings", expanded=False):
        embedding_model = st.selectbox("Embedding Model", ["all-MiniLM-L6-v2", "all-mpnet-base-v2"])
        energy_type = st.selectbox("Energy Function", ["softmax", "sparsemax", "polynomial"], help="Changes the Hopfield attention landscape.")
        use_projection = st.toggle("Learned Projection Layer", value=False, help="Uses InfoNCE-trained MLP to project queries into the Hopfield landscape.")
        hopfield_beta = st.slider("Temperature (Beta)", 0.1, 20.0, 8.0, 0.5)
        chunk_size = st.slider("Chunk Size", 128, 1024, 512, 64)
        chunk_overlap = st.slider("Chunk Overlap", 0, 256, 64, 16)
        top_k = st.slider("Retrieve Top-K", 1, 15, 5)

    pipeline = get_pipeline(embedding_model, chunk_size, chunk_overlap, hopfield_beta, top_k, energy_type, use_projection)

    st.subheader("2. Multimodal Knowledge Base")
    
    # Document Upload Supporting Multimodal Files
    supported_types = ["txt", "pdf", "md", "docx", "pptx", "xlsx", "csv", "png", "jpg", "jpeg", "webp"]
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, DOCX, PPTX, XLSX, CSV, Images, TXT)",
        type=supported_types,
        accept_multiple_files=True
    )
    if st.button("Ingest Files", use_container_width=True) and uploaded_files:
        reader = st.session_state.multimodal_reader
        with st.spinner("Parsing multimodal documents..."):
            for f in uploaded_files:
                try:
                    file_bytes = f.read()
                    parsed = reader.parse_file(file_bytes, f.name)
                    doc_text = parsed.full_text
                    if not doc_text or not doc_text.strip():
                        from utils.file_utils import extract_text_from_file
                        f.seek(0)
                        doc_text, _ = extract_text_from_file(f, f.name)
                    pipeline.add_document(doc_text, source=f.name)
                except Exception as e:
                    st.error(f"Failed to process {f.name}: {e}")
            st.session_state.docs_loaded = True
            st.success("Documents successfully ingested into Hopfield Associative Memory!")
            st.rerun()

    # Custom Text
    with st.expander("Paste Text Manually"):
        custom_name = st.text_input("Source Name", value="Pasted Snippet")
        custom_text = st.text_area("Content")
        if st.button("Add Text", use_container_width=True) and custom_text:
            with st.spinner("Processing..."):
                pipeline.add_document(custom_text.strip(), source=custom_name)
                st.session_state.docs_loaded = True
                st.success("Text added!")
                st.rerun()
                
    st.divider()
    if st.session_state.docs_loaded:
        st.metric("Stored Patterns (Chunks)", pipeline.n_chunks)
        st.markdown("**Active Sources:**")
        for src in pipeline._documents.keys():
            st.markdown(f"<span class='doc-tag'>📄 {src}</span>", unsafe_allow_html=True)
        
        if st.button("Clear Memory", type="primary", use_container_width=True):
            st.session_state.pipeline = None
            st.session_state.docs_loaded = False
            st.session_state.messages = []
            st.rerun()

# ---------------------------------------------------------
# MAIN INTERFACE (Chat)
# ---------------------------------------------------------
st.info(
    "💡 **Full Multimodal Knowledge Suite**: The production web application includes "
    "interactive Visual Studio diagrams, AI flashcard decks, hierarchical memory diagnostics, "
    "and knowledge graph explorers. Start `python api.py` to access it at [http://localhost:8000](http://localhost:8000)."
)

if not st.session_state.docs_loaded:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #4B5563;'>Continuous Modern Hopfield Associative QA</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #9CA3AF;'>Upload documents (PDF, DOCX, PPTX, XLSX, CSV, Images) in the sidebar to begin associative exploration.</p>", unsafe_allow_html=True)
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "results" in msg:
                with st.expander("🔍 View Hopfield Retrieval Details"):
                    for idx, chunk in enumerate(msg["results"]):
                        sim = getattr(chunk, 'similarity_score', getattr(chunk, 'hopfield_attention', 0.0))
                        src = getattr(chunk, 'source', 'Document')
                        txt = getattr(chunk, 'text', '')
                        st.markdown(f"**Rank {idx+1}** | Source: `{src}` | Score: `{float(sim):.4f}`")
                        st.caption(f"_{txt}_")
                        st.divider()

    # Chat Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Converging continuous energy state & synthesizing answer..."):
                start_t = time.time()
                try:
                    res = pipeline.query(
                        raw_query=prompt,
                        top_k=top_k,
                        retriever_mode="hopfield",
                        answer_style="conversational"
                    )
                    ans_text = res.answer.answer if hasattr(res.answer, 'answer') else str(res.answer)
                    confidence = getattr(res.answer, 'confidence', 1.0)
                    retrieved_chunks = res.evidence_collection.chunks if (res.evidence_collection and res.evidence_collection.chunks) else res.retrieved_chunks
                except Exception:
                    # Fallback to extractive answer generation
                    legacy_res = pipeline.generate_answer(prompt)
                    ans_text = legacy_res.answer.answer
                    confidence = legacy_res.answer.confidence
                    retrieved_chunks = legacy_res.retrieved_chunks

                elapsed = (time.time() - start_t) * 1000
                response_md = f"{ans_text}\n\n*Confidence: {confidence:.2f} | Time: {elapsed:.0f}ms*"
                st.markdown(response_md)
                
                with st.expander("🔍 View Hopfield Retrieval Details"):
                    for idx, chunk in enumerate(retrieved_chunks):
                        sim = getattr(chunk, 'similarity_score', getattr(chunk, 'hopfield_attention', 0.0))
                        src = getattr(chunk, 'source', 'Document')
                        txt = getattr(chunk, 'text', '')
                        st.markdown(f"**Rank {idx+1}** | Source: `{src}` | Score: `{float(sim):.4f}`")
                        st.caption(f"_{txt}_")
                        st.divider()
                        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_md,
            "results": retrieved_chunks
        })
