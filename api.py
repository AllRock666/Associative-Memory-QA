from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import os
import io
import tempfile
import traceback
from typing import List, Optional, Dict, Any

from core.pipeline import HopfieldQAPipeline, PipelineConfig
from core.llm_provider import LLMConfig, create_llm_provider, MockLLMProvider
from core.chat_manager import ChatManager
from utils.file_utils import extract_text_from_file
from core.multimodal_reader import MultimodalDocumentReader, MultimodalDocument
from core.visual_studio import VisualStudioEngine
from core.study_assistant import StudyAssistantEngine
from core.hopfield_hierarchical import HierarchicalHopfieldMemory
import base64
import csv
from fastapi.responses import Response


pipeline = None
chat_manager = ChatManager("data")
multimodal_reader = MultimodalDocumentReader()
multimodal_store: Dict[str, MultimodalDocument] = {}
visual_studio = VisualStudioEngine()
study_assistant = StudyAssistantEngine()
hierarchical_memory: Optional[HierarchicalHopfieldMemory] = None

class VisualStudioRequest(BaseModel):
    prompt: str
    requested_type: Optional[str] = None
    doc_id: Optional[str] = None
    page_num: Optional[int] = None

class StudyDeckRequest(BaseModel):
    topic: Optional[str] = "General Knowledge"
    doc_id: Optional[str] = None

class LLMVerifyRequest(BaseModel):
    provider: str = "ollama"
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    energy_type: Optional[str] = "softmax"
    hopfield_beta: Optional[float] = 8.0
    top_k: Optional[int] = 5
    use_projection: Optional[bool] = False
    retriever_mode: Optional[str] = "hopfield"  # "hopfield", "dense", "hybrid"
    engine_mode: Optional[str] = "associative"  # legacy backwards compatibility
    answer_style: Optional[str] = "conversational"  # "conversational" (LLM RAG), "detailed", "executive", "concise"
    detail_level: Optional[str] = None  # "adaptive", "detailed", "standard", "concise"
    llm_provider: Optional[str] = None  # "ollama", "openai", "mock", "deterministic"
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    max_tokens: Optional[int] = 8192
    temperature: Optional[float] = 0.2
    chat_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None
    web_search_enabled: Optional[bool] = True
    force_web_search: Optional[bool] = False
    web_search_provider: Optional[str] = None
    answer_format: Optional[str] = None
    active_doc_id: Optional[str] = None

class CreateChatRequest(BaseModel):
    title: Optional[str] = None
    chat_id: Optional[str] = None

class UpdateChatRequest(BaseModel):
    title: str

def init_app_state():
    global pipeline, hierarchical_memory
    if pipeline is not None:
        return
    proj_path = "models/projection_squad.pt" if os.path.exists("models/projection_squad.pt") else None
    llm_env = LLMConfig.from_env()
    web_search_active = os.environ.get("WEB_SEARCH_ENABLED", "true").lower() not in ("false", "0", "no")
    config = PipelineConfig(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=512,
        chunk_overlap=64,
        hopfield_beta=8.0,
        top_k=5,
        use_fusion=True,
        use_projection=False,
        projection_path=proj_path,
        use_associative_engine=True,
        web_search_enabled=web_search_active,
        web_search_provider=os.environ.get("WEB_SEARCH_PROVIDER", "composite"),
        llm_provider=llm_env.provider,
        llm_base_url=llm_env.base_url,
        llm_model=llm_env.model,
        llm_api_key=llm_env.api_key,
    )
    pipeline = HopfieldQAPipeline(config)
    hierarchical_memory = HierarchicalHopfieldMemory(embedding_dim=pipeline.embedder.embedding_dim)
    pipeline.hierarchical_memory = hierarchical_memory
    if hasattr(pipeline, "answer_generator") and hasattr(pipeline.answer_generator, "llm_provider"):
        visual_studio.llm_provider = pipeline.answer_generator.llm_provider

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_app_state()
    yield

app = FastAPI(title="Hopfield QA API", lifespan=lifespan)

def startup_event():
    """Backwards compatibility alias for tests and direct callers."""
    init_app_state()

@app.get("/api/status")
def get_status():
    global pipeline
    if not pipeline:
        return {"ready": False, "documents": 0, "chunks": 0, "llm": {"available": False}}

    is_avail = False
    avail_reason = None
    if hasattr(pipeline.answer_generator, "llm_provider") and pipeline.answer_generator.llm_provider:
        is_avail, avail_reason = pipeline.answer_generator.llm_provider.is_available()

    llm_status = {
        "provider": pipeline.config.llm_provider,
        "configured_provider": pipeline.config.llm_provider,
        "effective_provider": pipeline.config.llm_provider,
        "model": pipeline.config.llm_model,
        "base_url": pipeline.config.llm_base_url,
        "fallback_to_deterministic": pipeline.config.llm_fallback_to_deterministic,
        "fallback_enabled": pipeline.config.llm_fallback_to_deterministic,
        "available": is_avail,
        "is_healthy": is_avail,
        "status_message": avail_reason or "Provider is healthy and reachable.",
        "fallback_reason": avail_reason if not is_avail else None
    }

    return {
        "ready": pipeline.is_ready,
        "documents": pipeline.n_documents,
        "chunks": pipeline.n_chunks,
        "llm": llm_status,
    }

@app.post("/api/llm/verify")
def verify_llm_connection(req: LLMVerifyRequest):
    """
    Ephemerally verify connectivity and credentials for an LLM provider.
    Never persists or logs the API key to disk.
    """
    provider_name = (req.provider or "ollama").lower().strip()

    if provider_name in ("offline", "offline_grounded", "deterministic"):
        return {
            "success": True,
            "provider": "offline",
            "model": "deterministic_grounded",
            "message": "Offline Grounded Mode active. 100% private, deterministic Hopfield extraction."
        }
    
    if provider_name == "groq":
        base_url = (req.base_url or "https://api.groq.com/openai/v1").strip()
        model = (req.model or "llama-3.3-70b-versatile").strip()
    elif provider_name == "openai":
        base_url = (req.base_url or "https://api.openai.com/v1").strip()
        model = (req.model or "gpt-4o-mini").strip()
    elif provider_name == "gemini":
        base_url = (req.base_url or "https://generativelanguage.googleapis.com/v1beta/openai/").strip()
        model = (req.model or "gemini-1.5-flash").strip()
    elif provider_name == "ollama":
        base_url = (req.base_url or "http://localhost:11434/v1").strip()
        model = (req.model or "llama3.2").strip()
    else:
        base_url = (req.base_url or "https://api.openai.com/v1").strip()
        model = (req.model or "gpt-4o-mini").strip()

    if provider_name == "ollama":
        root_url = base_url.replace("/v1", "").rstrip("/")
        tags_url = f"{root_url}/api/tags"
        try:
            import httpx
            with httpx.Client(timeout=httpx.Timeout(4.0, connect=2.0)) as client:
                resp = client.get(tags_url)
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    model_names_str = ", ".join(models[:4]) if models else "none downloaded yet"
                    return {
                        "success": True,
                        "provider": "ollama",
                        "model": model,
                        "message": f"Connected to local Ollama successfully! Found models: {model_names_str}."
                    }
                return {
                    "success": False,
                    "provider": "ollama",
                    "message": f"Ollama returned HTTP status {resp.status_code}."
                }
        except Exception as e:
            return {
                "success": False,
                "provider": "ollama",
                "message": f"Could not connect to Ollama at {root_url}: {str(e)}. Make sure Ollama is installed and running."
            }

    api_key = (req.api_key or "").strip()
    if not api_key:
        return {
            "success": False,
            "provider": provider_name,
            "message": "API Key is required for external cloud LLM providers."
        }

    try:
        import httpx
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        test_payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Respond with OK"}],
            "max_tokens": 5,
            "temperature": 0.0
        }
        with httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
            resp = client.post(url, json=test_payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                return {
                    "success": True,
                    "provider": provider_name,
                    "model": model,
                    "message": f"Verified successfully with {provider_name.upper()} ({model})! Live response: '{reply}'."
                }
            elif resp.status_code in (401, 403):
                return {
                    "success": False,
                    "provider": provider_name,
                    "message": f"Authentication failed (HTTP {resp.status_code}): Invalid API key or unauthorized."
                }
            elif resp.status_code == 404:
                return {
                    "success": False,
                    "provider": provider_name,
                    "message": f"HTTP 404: Model '{model}' not found or endpoint URL '{url}' is invalid."
                }
            else:
                err_text = resp.text[:200]
                return {
                    "success": False,
                    "provider": provider_name,
                    "message": f"Provider returned HTTP {resp.status_code}: {err_text}"
                }
    except Exception as e:
        return {
            "success": False,
            "provider": provider_name,
            "message": f"Connection error reaching {base_url}: {str(e)}"
        }

@app.post("/api/search")
def search(req: SearchRequest):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
    pipeline.config.energy_type = req.energy_type
    pipeline.config.hopfield_beta = req.hopfield_beta
    pipeline.config.top_k = req.top_k or 5
    pipeline.config.use_projection = req.use_projection
    pipeline.memory.retriever.beta = req.hopfield_beta
    pipeline.memory.retriever.energy_type = req.energy_type
    pipeline.memory.retriever.use_projection = req.use_projection
    
    # Determine retriever mode
    active_retriever = (req.retriever_mode or "hopfield").lower()
    if req.engine_mode == "naive_rag" and req.retriever_mode == "hopfield":
        active_retriever = "dense"
        
    style = req.detail_level or req.answer_style or "conversational"

    # Optional dynamic LLM provider override
    provider_override = None
    if req.llm_provider:
        p_name = req.llm_provider.lower().strip()
        ovr_cfg = LLMConfig.from_env()
        ovr_cfg.provider = p_name
        if p_name == "groq" and not req.llm_base_url:
            ovr_cfg.base_url = "https://api.groq.com/openai/v1"
            if not req.llm_model:
                ovr_cfg.model = "llama-3.3-70b-versatile"
        elif p_name == "openai" and not req.llm_base_url:
            ovr_cfg.base_url = "https://api.openai.com/v1"
            if not req.llm_model:
                ovr_cfg.model = "gpt-4o-mini"
        elif p_name == "gemini" and not req.llm_base_url:
            ovr_cfg.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            if not req.llm_model:
                ovr_cfg.model = "gemini-1.5-flash"
        elif p_name == "ollama" and not req.llm_base_url:
            ovr_cfg.base_url = "http://localhost:11434/v1"
            if not req.llm_model:
                ovr_cfg.model = "llama3.2"

        if req.llm_model:
            ovr_cfg.model = req.llm_model
        if req.llm_base_url:
            ovr_cfg.base_url = req.llm_base_url
        if req.llm_api_key:
            ovr_cfg.api_key = req.llm_api_key
        if req.max_tokens:
            ovr_cfg.max_tokens = req.max_tokens
        if req.temperature is not None:
            ovr_cfg.temperature = req.temperature
        provider_override = create_llm_provider(ovr_cfg)
    
    # Extract conversation history for multi-turn conversational context
    history = req.conversation_history
    if not history and req.chat_id:
        chat_sess = chat_manager.get_session(req.chat_id)
        if chat_sess:
            history = [{"role": m["role"], "content": m["content"]} for m in chat_sess.get("messages", [])]

    # Adaptive retrieval depth: scale k for detailed, multi-aspect, or explanatory questions
    q_lower = req.query.lower()
    q_words = req.query.strip().split()

    # Automatically index any large paragraphs pasted directly in history or current prompt
    if history:
        for idx, h_msg in enumerate(history):
            h_text = h_msg.get("content", "").strip()
            if h_msg.get("role") == "user" and len(h_text) > 120 and len(h_text.split()) > 15:
                h_src = f"pasted_chat_{req.chat_id or 'turn'}_{idx}"
                if h_src not in pipeline._documents:
                    pipeline.add_documents([(h_text, h_src)])
                    if hierarchical_memory:
                        chunks = [{"text": h_text, "source": h_src, "page_number": 1, "element_type": "text"}]
                        emb = pipeline.embedder.embed_texts([h_text], show_progress=False)
                        hierarchical_memory.ingest_chunks(chunks, emb)

    if len(req.query.strip()) > 150 and (len(req.query.split()) > 20 or "\n" in req.query):
        curr_src = f"pasted_chat_{req.chat_id or 'current'}"
        if curr_src not in pipeline._documents:
            pipeline.add_documents([(req.query.strip(), curr_src)])
            if hierarchical_memory:
                chunks = [{"text": req.query.strip(), "source": curr_src, "page_number": 1, "element_type": "text"}]
                emb = pipeline.embedder.embed_texts([req.query.strip()], show_progress=False)
                hierarchical_memory.ingest_chunks(chunks, emb)

    is_multi_aspect = any(w in q_lower for w in [
        "explain", "architecture", "components", "mechanism", "mechanisms", "responsibilities",
        "difference", "compare", "breakdown", "elaborate", "comprehensive", "lifecycle",
        "and how", "and what", "overview", "workflow", "workloads", "detail", "detailed",
        "how does", "why does"
    ]) or len(q_words) >= 6

    user_k = req.top_k
    if user_k and user_k > 5:
        effective_k = user_k
    elif style in ("detailed", "comprehensive", "technical") or is_multi_aspect:
        effective_k = 12
    elif style == "executive" or any(w in q_lower for w in ["what is", "define", "meaning", "concept"]):
        effective_k = 8
    else:
        effective_k = 6

    try:
        res = pipeline.query(
            raw_query=req.query,
            top_k=effective_k,
            retriever_mode=active_retriever,
            answer_style=style,
            llm_provider_override=provider_override,
            conversation_history=history,
            web_search_enabled=req.web_search_enabled,
            force_web_search=bool(req.force_web_search),
            web_search_provider=req.web_search_provider,
            answer_format=req.answer_format,
            active_doc_id=req.active_doc_id,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    
    evidence_list = []
    source_chunks = res.evidence_collection.chunks if (res.evidence_collection and res.evidence_collection.chunks) else res.retrieved_chunks
    for i, chunk in enumerate(source_chunks, 1):
        sim = getattr(chunk, 'similarity_score', getattr(chunk, 'hopfield_attention', 0.0))
        meta = getattr(chunk, 'metadata', {}) or {}
        is_core = getattr(chunk, 'is_core_attractor', meta.get("is_core_attractor", True))
        evidence_list.append({
            "citation_id": f"E{i}",
            "chunk_id": getattr(chunk, 'chunk_id', f"chunk_{i}"),
            "text": getattr(chunk, 'text', ''),
            "score": float(sim) if sim is not None else 0.0,
            "source": getattr(chunk, 'source', 'Unknown'),
            "source_type": getattr(chunk, 'source_type', meta.get('source_type', 'document')),
            "url": getattr(chunk, 'url', meta.get('url')),
            "title": getattr(chunk, 'title', meta.get('title')),
            "published_date": getattr(chunk, 'published_date', meta.get('published_date')),
            "page_number": getattr(chunk, 'page_number', meta.get("page_number")),
            "is_core_attractor": is_core,
            "role": "Core Attractor Basin" if is_core else "Associative Bridge"
        })
    
    associative_diagnostics = None
    if res.associative_result:
        ar = res.associative_result
        trajectory_data = []
        for hop in ar.trajectory:
            trajectory_data.append({
                "hop": hop.hop_index,
                "energy": round(hop.energy, 4),
                "delta": round(hop.state_delta, 4),
                "description": hop.description,
                "top_chunks": hop.activated_chunks[:2]
            })
        associative_diagnostics = {
            "mode": f"Hopfield ({res.evidence_collection.retriever_name if res.evidence_collection else 'Associative'})",
            "attractor_energy": round(ar.attractor_energy, 4),
            "energy_depth": round(ar.energy_landscape_depth, 4),
            "is_grounded": ar.is_grounded,
            "convergence_steps": ar.convergence_steps,
            "reasoning_path": ar.reasoning_path,
            "trajectory": trajectory_data,
            "num_core_chunks": len(ar.core_attractor_chunks),
            "num_bridge_chunks": len(ar.associative_bridge_chunks)
        }
    else:
        associative_diagnostics = {
            "mode": f"{res.evidence_collection.retriever_name if res.evidence_collection else 'Dense Cosine'}",
            "attractor_energy": None,
            "energy_depth": None,
            "is_grounded": len(evidence_list) > 0,
            "convergence_steps": 1,
            "reasoning_path": ["Static dot product ranking without continuous associative relaxation."],
            "trajectory": [],
            "num_core_chunks": len(evidence_list),
            "num_bridge_chunks": 0
        }

    grounded_data = None
    if res.grounded_answer:
        ga = res.grounded_answer
        grounded_data = {
            "answer": ga.answer,
            "abstained": ga.abstained,
            "abstention_reason": ga.abstention_reason,
            "confidence": round(ga.confidence, 4),
            "claims": [
                {
                    "claim_text": c.claim_text,
                    "evidence_ids": c.evidence_ids,
                    "is_verified": c.is_verified,
                    "unresolved_ids": c.unresolved_ids,
                    "is_supported": getattr(c, 'is_supported', c.is_verified),
                    "is_relevant": getattr(c, 'is_relevant', True),
                    "is_unique": getattr(c, 'is_unique', True),
                    "verification_rationale": getattr(c, 'verification_rationale', None)
                } for c in ga.claims
            ],
            "provider_used": ga.provider_used,
            "model_name": ga.model_name,
            "fallback_occurred": ga.metadata.get("fallback_occurred", False),
            "fallback_reason": ga.metadata.get("fallback_reason"),
            "llm_config": ga.metadata.get("llm_config", {}),
            "generation_time_ms": max(0.1, round(ga.generation_time_ms, 2)),
            "style": ga.style,
            "stage_latencies_ms": ga.metadata.get("stage_latencies_ms", {}),
            "confidence_breakdown": ga.metadata.get("confidence_breakdown", {}),
            "deduplication": getattr(res, "deduplication_diagnostics", {}) or ga.metadata.get("deduplication", {}),
            "unique_verified_claims_count": ga.metadata.get("unique_verified_claims_count", len([c for c in ga.claims if c.is_verified])),
            "total_claims_count": len(ga.claims)
        }
    
    latencies = {
        "retrieval_ms": round(res.retrieval_time_ms, 2),
        "total_ms": round(res.total_time_ms, 2),
        "stages": getattr(res, "stage_latencies_ms", {})
    }

    contextualized_q = getattr(getattr(res, "processed_query", None), "metadata", {}).get("contextualized_query")
    response_payload = {
        "query": req.query,
        "contextualized_query": contextualized_q,
        "retriever_mode": active_retriever,
        "engine_mode": "associative" if active_retriever in ("hopfield", "hybrid") else "naive_rag",
        "answer": res.answer.answer,
        "confidence": float(res.answer.confidence) if res.answer.confidence is not None else 0.0,
        "evidence": evidence_list,
        "grounded_answer": grounded_data,
        "associative_diagnostics": associative_diagnostics,
        "latencies": latencies,
        "deduplication": getattr(res, "deduplication_diagnostics", {}),
        "web_search_used": getattr(res, "web_search_used", False),
        "cached_knowledge_used": getattr(res, "cached_knowledge_used", False),
        "format_detected": getattr(res, "format_detected", None),
        "format_requested": req.answer_format or req.detail_level or req.answer_style,
        "sufficiency_decision": getattr(res, "sufficiency_decision", None),
        "web_sources": getattr(res, "web_sources", [])
    }

    # Check if this query requested a visualization (e.g. mind map, flowchart, architecture diagram)
    is_viz_query = any(w in req.query.lower() for w in ["mind map", "mindmap", "flowchart", "flow chart", "architecture diagram", "sequence diagram", "concept map", "draw a diagram", "make a diagram"])
    if is_viz_query:
        # Pass active LLM provider to VisualStudioEngine if available
        if provider_override:
            visual_studio.llm_provider = provider_override
        elif hasattr(pipeline, "answer_generator") and hasattr(pipeline.answer_generator, "llm_provider"):
            visual_studio.llm_provider = pipeline.answer_generator.llm_provider

        # Resolve the most relevant source text for the visualization:
        found_target_text = None
        raw_q = req.query.strip()

        # 1. Did the user paste the paragraph directly inside the visualization request?
        if len(raw_q) > 120 and len(raw_q.split()) > 15:
            clean_q_text = re.sub(r'^(?:please\s+)?(?:make|create|generate|build|draw)\s+(?:a\s+)?(?:mind\s*map|diagram|flowchart|concept\s*map)(?:\s+(?:of|for|about|from))?\s*[:\-–—]?\s*', '', raw_q, flags=re.I).strip()
            if len(clean_q_text) > 80:
                found_target_text = clean_q_text
            else:
                found_target_text = raw_q

        # 2. Check conversation history for a recently pasted paragraph or substantial user text
        if not found_target_text and history:
            for h in reversed(history):
                if h.get("role") == "user":
                    c_text = h.get("content", "").strip()
                    if c_text != raw_q and len(c_text) > 80 and len(c_text.split()) > 10:
                        found_target_text = c_text
                        break

        # 3. Check active multimodal document if one is selected
        if not found_target_text and req.active_doc_id and req.active_doc_id in multimodal_store:
            doc_obj = multimodal_store[req.active_doc_id]
            if doc_obj and getattr(doc_obj, "full_text", None):
                found_target_text = doc_obj.full_text

        # 4. Check pipeline documents (pasted chats or uploaded docs)
        if not found_target_text and pipeline and pipeline._documents:
            pasted_keys = [k for k in pipeline._documents.keys() if k.startswith("pasted_chat")]
            if pasted_keys:
                found_target_text = pipeline._documents[pasted_keys[-1]]
            else:
                latest_src = list(pipeline._documents.keys())[-1]
                found_target_text = pipeline._documents[latest_src]

        if found_target_text:
            viz_context = [{"text": found_target_text, "source": "Target Document/Paragraph", "page_number": 1}]
        else:
            viz_context = [
                {"text": c.get("text", ""), "source": c.get("source", "Document"), "page_number": c.get("page_number", 1)}
                for c in evidence_list
            ]

        viz_res = visual_studio.generate_visualization(
            prompt=req.query,
            context_chunks=viz_context
        )
        if viz_res.viz_type == "mermaid":
            mermaid_block = f"\n\n```mermaid\n{viz_res.content}\n```\n\n*{viz_res.description}*"
            if response_payload["answer"].startswith("I cannot answer"):
                response_payload["answer"] = f"### Conceptual Mind Map\n\n```mermaid\n{viz_res.content}\n```\n\n*{viz_res.description}*"
            else:
                response_payload["answer"] += mermaid_block
        response_payload["visualization"] = {
            "viz_type": viz_res.viz_type,
            "subtype": viz_res.subtype,
            "title": viz_res.title,
            "content": viz_res.content
        }

    # Automatically persist turn to chat session if chat_id provided
    if req.chat_id:
        chat_sess = chat_manager.add_turn(req.chat_id, req.query, response_payload)
        response_payload["chat_id"] = req.chat_id
        response_payload["chat_title"] = chat_sess.get("title")

    return response_payload


# ==============================================================================
# Multi-Chat Persistence Endpoints (ChatGPT-Style Sessions)
# ==============================================================================

@app.get("/api/chats")
def list_chats():
    """List all persisted chat conversations."""
    return {"chats": chat_manager.list_sessions()}


@app.post("/api/chats")
def create_or_save_chat(req: Optional[CreateChatRequest] = None):
    """Create a new chat conversation."""
    title = req.title if req else None
    c_id = req.chat_id if req else None
    session = chat_manager.create_session(title=title, session_id=c_id)
    return {"chat": session}


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str):
    """Retrieve full conversation history for a specific chat."""
    session = chat_manager.get_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"chat": session}


@app.put("/api/chats/{chat_id}")
def update_chat_title(chat_id: str, req: UpdateChatRequest):
    """Update title of a chat conversation."""
    session = chat_manager.update_session(chat_id, title=req.title)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"chat": session}


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str):
    """Delete a chat conversation and remove from disk storage."""
    deleted = chat_manager.delete_session(chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"success": True, "message": f"Chat {chat_id} deleted"}


# ==============================================================================
# Persistent Documents / Knowledge Base Endpoints
# ==============================================================================

@app.get("/api/documents")
def list_documents():
    """List all documents currently active and persisted in Hopfield memory."""
    global pipeline
    if not pipeline:
        return {"documents": [], "total_documents": 0, "total_chunks": 0}
        
    chunk_counts = {}
    for c in pipeline._all_chunks:
        chunk_counts[c.source] = chunk_counts.get(c.source, 0) + 1
        
    docs = []
    for source, text in pipeline._documents.items():
        docs.append({
            "source": source,
            "chunks": chunk_counts.get(source, 0),
            "characters": len(text)
        })
    docs.sort(key=lambda d: d["source"].lower())
    return {
        "documents": docs,
        "total_documents": pipeline.n_documents,
        "total_chunks": pipeline.n_chunks
    }


@app.delete("/api/documents/{source:path}")
def remove_document(source: str):
    """Remove a document and its chunks from memory and persist changes."""
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
    removed = pipeline.remove_document(source)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Document '{source}' not found in memory")
        
    return {
        "success": True,
        "message": f"Successfully removed '{source}' from Hopfield memory",
        "remaining_documents": pipeline.n_documents,
        "remaining_chunks": pipeline.n_chunks
    }

@app.post("/api/upload")
def upload_document(file: UploadFile = File(...)):
    global pipeline, hierarchical_memory, multimodal_store
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB limit
    content = file.file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum upload limit of 25MB (size: {len(content) / (1024 * 1024):.1f}MB)."
        )

    try:
        # Parse through MultimodalDocumentReader
        parsed_doc = multimodal_reader.parse_file(content, file.filename)
        multimodal_store[parsed_doc.doc_id] = parsed_doc
        multimodal_store[file.filename] = parsed_doc

        text = parsed_doc.full_text
        if not text or not text.strip():
            bio = io.BytesIO(content)
            text, _ = extract_text_from_file(bio, file.filename)

        if not text or not text.strip():
            raise ValueError(f"Extracted text from '{file.filename}' is empty or unreadable.")

        print(f"[API] Upload request received: {file.filename} ({len(content)} bytes)", flush=True)
        docs = [(text, file.filename)]
        pipeline.add_documents(docs)

        # Ingest into Hierarchical Hopfield Memory 2.0
        if hierarchical_memory:
            chunks = parsed_doc.to_chunks()
            if chunks:
                chunk_texts = [c["text"] for c in chunks]
                embeddings = pipeline.embedder.embed_texts(chunk_texts, show_progress=False)
                hierarchical_memory.ingest_chunks(chunks, embeddings)

        total_chunks = pipeline.n_chunks
        print(f"[API] Successfully ingested {file.filename}: {total_chunks} total chunks indexed.", flush=True)
        return {
            "message": f"Successfully ingested {file.filename} ({total_chunks} chunks indexed in Hopfield memory).",
            "doc_id": parsed_doc.doc_id,
            "filename": file.filename,
            "total_pages": parsed_doc.total_pages,
            "tables_found": len(parsed_doc.tables),
            "diagrams_found": len(parsed_doc.diagrams)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


# ==============================================================================
# Multimodal Document Reader Endpoints
# ==============================================================================

@app.get("/api/documents/multimodal")
def list_multimodal_documents():
    """List parsed multimodal documents with page counts and structural stats."""
    docs = []
    seen = set()
    for doc_id, doc in multimodal_store.items():
        if doc.doc_id in seen:
            continue
        seen.add(doc.doc_id)
        docs.append({
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "total_pages": doc.total_pages,
            "tables_count": len(doc.tables),
            "diagrams_count": len(doc.diagrams)
        })
    return {"documents": docs, "total_documents": len(docs)}


@app.get("/api/documents/{doc_id}/pages")
def get_document_pages(doc_id: str):
    """Retrieve metadata and summaries for each page of a multimodal document."""
    doc = multimodal_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    pages_summary = []
    for p in doc.pages:
        pages_summary.append({
            "page_number": p.page_number,
            "snippet": p.text_content[:200] + "..." if len(p.text_content) > 200 else p.text_content,
            "has_tables": p.has_tables,
            "has_images": p.has_images,
            "ai_summary": p.ai_summary,
            "key_concepts": p.key_concepts
        })
    return {"doc_id": doc_id, "filename": doc.filename, "pages": pages_summary}


@app.get("/api/documents/{doc_id}/pages/{page_num}")
def get_page_details(doc_id: str, page_num: int):
    """Retrieve detailed elements and optional rendered image for a specific page."""
    doc = multimodal_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    page = doc.get_page(page_num)
    if not page:
        raise HTTPException(status_code=404, detail=f"Page {page_num} not found")

    img_b64 = None
    if page.page_image_bytes:
        img_b64 = base64.b64encode(page.page_image_bytes).decode("utf-8")

    elements_data = []
    for elem in page.elements:
        elements_data.append({
            "element_id": elem.element_id,
            "element_type": elem.element_type,
            "content": elem.content,
            "bbox": elem.bbox,
            "structured_data": elem.structured_data
        })

    return {
        "doc_id": doc_id,
        "page_number": page_num,
        "text_content": page.text_content,
        "ai_summary": page.ai_summary,
        "key_concepts": page.key_concepts,
        "has_tables": page.has_tables,
        "has_images": page.has_images,
        "image_base64": img_b64,
        "elements": elements_data
    }


@app.get("/api/documents/{doc_id}/tables")
def get_document_tables(doc_id: str):
    """List all extracted tables for a document."""
    doc = multimodal_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"doc_id": doc_id, "tables": doc.tables}


@app.get("/api/documents/{doc_id}/tables/{table_id}/csv")
def export_table_csv(doc_id: str, table_id: str):
    """Exports an extracted table as a downloadable CSV file."""
    doc = multimodal_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    target_table = None
    for tbl in doc.tables:
        if tbl.get("table_id") == table_id:
            target_table = tbl
            break
    if not target_table:
        raise HTTPException(status_code=404, detail="Table not found")

    output = io.StringIO()
    writer = csv.writer(output)
    if target_table.get("headers"):
        writer.writerow(target_table["headers"])
    for row in target_table.get("rows", []):
        writer.writerow(row)
    csv_str = output.getvalue()

    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={table_id}.csv"}
    )


# ==============================================================================
# AI Visual Studio Endpoints
# ==============================================================================

@app.post("/api/visual-studio/generate")
def generate_studio_visualization(req: VisualStudioRequest):
    """
    Generates Mermaid diagrams, Plotly interactive charts, or Cytoscape graphs
    grounded in retrieved Hopfield memory.
    """
    global pipeline, hierarchical_memory
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    # Retrieve context chunks
    q_emb = pipeline.embedder.embed_query(req.prompt)
    if hierarchical_memory:
        retrieval = hierarchical_memory.retrieve(req.prompt, q_emb, top_k=6)
        context_chunks = retrieval.core_attractors + retrieval.associative_bridges
    else:
        results = pipeline.memory.retrieve(q_emb, top_k=6)
        context_chunks = [{"text": r.text, "source": r.source, "page_number": getattr(r, "metadata", {}).get("page", 1)} for r in results]

    viz_result = visual_studio.generate_visualization(
        prompt=req.prompt,
        context_chunks=context_chunks,
        requested_type=req.requested_type
    )

    return {
        "viz_type": viz_result.viz_type,
        "subtype": viz_result.subtype,
        "title": viz_result.title,
        "description": viz_result.description,
        "content": viz_result.content,
        "raw_data": viz_result.raw_data,
        "citations": viz_result.citations,
        "has_numerical_data": viz_result.has_numerical_data,
        "suggested_alternative": viz_result.suggested_alternative
    }


# ==============================================================================
# Knowledge Graph Explorer Endpoints
# ==============================================================================

@app.get("/api/knowledge-graph")
def get_knowledge_graph(max_nodes: int = 50):
    """Returns Cytoscape.js compatible graph data from Level 4 memory."""
    global hierarchical_memory
    if not hierarchical_memory:
        return {"elements": {"nodes": [], "edges": []}}
    return hierarchical_memory.knowledge_graph.to_cytoscape(max_nodes=max_nodes)


# ==============================================================================
# AI Study Assistant Endpoints
# ==============================================================================

@app.post("/api/study/deck")
def create_study_deck(req: StudyDeckRequest):
    """Generates flashcards, MCQs, and revision roadmap for active learning."""
    global pipeline, hierarchical_memory
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    topic = req.topic or "Document Concepts"
    q_emb = pipeline.embedder.embed_query(topic)
    if hierarchical_memory:
        retrieval = hierarchical_memory.retrieve(topic, q_emb, top_k=6)
        context_chunks = retrieval.core_attractors + retrieval.associative_bridges
    else:
        results = pipeline.memory.retrieve(q_emb, top_k=6)
        context_chunks = [{"text": r.text, "source": r.source, "page_number": getattr(r, "metadata", {}).get("page", 1)} for r in results]

    deck = study_assistant.generate_study_deck(topic, context_chunks)
    return deck.to_dict()


# ==============================================================================
# Hopfield Hierarchical Memory & Benchmark Endpoints
# ==============================================================================

@app.get("/api/memory/hierarchical/stats")
def get_memory_stats():
    """Returns 5-level memory health and diagnostic metrics."""
    global hierarchical_memory
    if not hierarchical_memory:
        return {"error": "Hierarchical memory not initialized"}
    return hierarchical_memory.get_stats()


@app.get("/api/memory/benchmark")
def run_benchmark():
    """Runs standard Recall, MRR, NDCG, and energy basin benchmarks."""
    from eval.benchmark_hopfield import run_hopfield_benchmark
    return run_hopfield_benchmark()


# ==============================================================================
# Answer Formats & Exam Rubrics Endpoints
# ==============================================================================

@app.get("/api/formats")
def list_formats():
    """List available answer formats and academic/technical mark rubrics."""
    from core.format_guidance import FORMAT_CATALOG
    catalog_list = []
    for fid, g in FORMAT_CATALOG.items():
        catalog_list.append({
            "format_id": fid,
            "name": getattr(g, "name", g.display_name),
            "display_name": g.display_name,
            "target_words": g.expected_word_range,
            "mark_weight": g.target_marks,
            "description": g.expected_word_range,
            "outline": g.structural_outline,
            "target_components": g.target_components
        })
    return {"formats": catalog_list}


# ==============================================================================
# Persistent Web Knowledge Memory Endpoints
# ==============================================================================

@app.get("/api/knowledge")
def list_knowledge():
    """List cached persistent web knowledge memory entries."""
    global pipeline
    if not pipeline or not hasattr(pipeline, "knowledge_memory"):
        return {"entries": [], "total_entries": 0}
    entries = pipeline.knowledge_memory.get_all_entries()
    return {
        "entries": [e.to_dict() for e in entries],
        "total_entries": len(entries)
    }


@app.delete("/api/knowledge")
def clear_knowledge():
    """Clear all persistent web knowledge memory entries."""
    global pipeline
    if not pipeline or not hasattr(pipeline, "knowledge_memory"):
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    pipeline.knowledge_memory.clear()
    return {"success": True, "message": "Persistent knowledge memory cleared"}


@app.delete("/api/knowledge/{entry_id}")
def delete_knowledge_entry(entry_id: str):
    """Delete a specific persistent knowledge memory entry."""
    global pipeline
    if not pipeline or not hasattr(pipeline, "knowledge_memory"):
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    found = pipeline.knowledge_memory.delete_entry(entry_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Knowledge entry '{entry_id}' not found")
    return {"success": True, "message": f"Knowledge entry '{entry_id}' deleted"}


@app.get("/")
async def get_index():
    response = FileResponse("static/index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
