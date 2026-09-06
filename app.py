import sys
import os
import time
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------
# PAGE CONFIG & CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hopfield Search",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Clean Minimalist Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .stChatFloatingInputContainer {
        padding-bottom: 2rem;
    }
    
    /* Clean metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
    }
    
    /* Subtle expanders */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #4B5563;
    }
    
    /* Document tags */
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

init_state()

# ---------------------------------------------------------
# PIPELINE INITIALIZATION
# ---------------------------------------------------------
def get_pipeline(embedding_model, chunk_size, chunk_overlap, hopfield_beta, top_k, energy_type):
    from core.pipeline import HopfieldQAPipeline, PipelineConfig
    
    if st.session_state.pipeline is None:
        config = PipelineConfig(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            hopfield_beta=hopfield_beta,
            top_k=top_k,
            use_fusion=True,
        )
        config.energy_type = energy_type
        st.session_state.pipeline = HopfieldQAPipeline(config)
    else:
        p = st.session_state.pipeline
        p.config.hopfield_beta = hopfield_beta
        p.config.top_k = top_k
        p.config.chunk_size = chunk_size
        p.config.energy_type = energy_type
        p.memory.retriever.beta = hopfield_beta
        p.memory.retriever.energy_type = energy_type
        
    return st.session_state.pipeline

# ---------------------------------------------------------
# SIDEBAR (Data & Config)
# ---------------------------------------------------------
with st.sidebar:
    st.title("🧠 Hopfield Search")
    st.markdown("Neural Associative Memory QA")
    st.divider()
    
    st.subheader("1. System Configuration")
    with st.expander("⚙️ Advanced Settings", expanded=False):
        embedding_model = st.selectbox("Embedding Model", ["all-MiniLM-L6-v2", "all-mpnet-base-v2"])
        energy_type = st.selectbox("Energy Function", ["softmax", "sparsemax", "polynomial"], help="Changes the Hopfield attention landscape.")
        hopfield_beta = st.slider("Temperature (Beta)", 0.1, 20.0, 8.0, 0.5)
        chunk_size = st.slider("Chunk Size", 128, 1024, 512, 64)
        chunk_overlap = st.slider("Chunk Overlap", 0, 256, 64, 16)
        top_k = st.slider("Retrieve Top-K", 1, 15, 3)

    pipeline = get_pipeline(embedding_model, chunk_size, chunk_overlap, hopfield_beta, top_k, energy_type)

    st.subheader("2. Knowledge Base")
    
    # Document Upload
    uploaded_files = st.file_uploader("Upload Documents (PDF, TXT)", type=["txt", "pdf", "md"], accept_multiple_files=True)
    if st.button("Ingest Files", use_container_width=True) and uploaded_files:
        from utils.file_utils import extract_text_from_file
        with st.spinner("Processing documents..."):
            for f in uploaded_files:
                try:
                    text, source = extract_text_from_file(f, f.name)
                    pipeline.add_document(text, source=source)
                except Exception as e:
                    st.error(f"Failed to process {f.name}: {e}")
            st.session_state.docs_loaded = True
            st.success("Documents added to Hopfield Memory!")
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
if not st.session_state.docs_loaded:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #6B7280;'>Welcome to Hopfield Search</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #9CA3AF;'>To get started, please upload some documents or paste text in the sidebar.</p>", unsafe_allow_html=True)
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "results" in msg:
                with st.expander("🔍 View Hopfield Retrieval Details"):
                    for idx, chunk in enumerate(msg["results"]):
                        st.markdown(f"**Rank {idx+1}** | Source: `{chunk.source}` | Attn: `{chunk.hopfield_attention:.4f}`")
                        st.caption(f"_{chunk.text}_")
                        st.divider()

    # Chat Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Converging energy state..."):
                start_t = time.time()
                result = pipeline.generate_answer(prompt)
                elapsed = (time.time() - start_t) * 1000
                
                ans_text = result.answer.answer
                
                response_md = f"{ans_text}\n\n*Confidence: {result.answer.confidence:.2f} | Time: {elapsed:.0f}ms*"
                st.markdown(response_md)
                
                with st.expander("🔍 View Hopfield Retrieval Details"):
                    for idx, chunk in enumerate(result.retrieved_chunks):
                        st.markdown(f"**Rank {idx+1}** | Source: `{chunk.source}` | Attn: `{chunk.hopfield_attention:.4f}`")
                        st.caption(f"_{chunk.text}_")
                        st.divider()
                        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_md,
            "results": result.retrieved_chunks
        })
