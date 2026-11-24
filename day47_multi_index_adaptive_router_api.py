import os
import time
import shutil
import chromadb
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(
    title="Day 47 Multi-Index Adaptive Router & Cost-Optimized RAG API",
    version="1.0.0",
    description="Query complexity classification, dynamic index dispatching, and latency-cost optimization"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day47_adaptive_router_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day47"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-Memory Exact Lexical Index (Tier 1)
LEXICAL_KEYWORD_INDEX: Dict[str, List[Dict[str, str]]] = {}
SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class AdaptiveRouterRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"
    max_latency_budget_ms: float = 800.0

class RoutingTelemetry(BaseModel):
    detected_complexity: str  # "SIMPLE_LOOKUP", "SEMANTIC_FACT", "DEEP_ANALYTIC"
    routed_tier: str          # "TIER_1_LEXICAL", "TIER_2_DENSE_VECTOR", "TIER_3_RERANKED_DEEP"
    routing_latency_ms: float
    retrieval_latency_ms: float
    synthesis_latency_ms: float
    total_pipeline_ms: float
    simulated_compute_cost_usd: float
    sla_met: bool

class AdaptiveRouterResponse(BaseModel):
    session_id: str
    standalone_question: str
    telemetry: RoutingTelemetry
    retrieved_evidence_summary: str
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 47 Adaptive Multi-Index Router RAG",
        "total_chunks_in_vector_index": collection.count(),
        "total_lexical_keywords": len(LEXICAL_KEYWORD_INDEX),
        "active_sessions": len(SESSION_HISTORIES)
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    reader = PdfReader(file_path)
    total_added = 0

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        words = text.split()
        chunk_size = 75
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        for c_idx, chunk in enumerate(chunks):
            doc_id = f"{file.filename}_p{page_idx + 1}_c{c_idx}"
            vector = embed_model.encode(chunk).tolist()

            # Index into Tier 2/3 Vector Store
            collection.add(
                documents=[chunk],
                embeddings=[vector],
                metadatas=[{"source": file.filename, "page": page_idx + 1}],
                ids=[doc_id]
            )
            total_added += 1

            # Populate Tier 1 Lexical Index
            for raw_word in set(words[i:i + chunk_size]):
                clean_term = "".join(filter(str.isalnum, raw_word)).lower()
                if len(clean_term) >= 4:
                    if clean_term not in LEXICAL_KEYWORD_INDEX:
                        LEXICAL_KEYWORD_INDEX[clean_term] = []
                    LEXICAL_KEYWORD_INDEX[clean_term].append({
                        "source": file.filename,
                        "page": str(page_idx + 1),
                        "chunk": chunk
                    })

    return {
        "status": "Success",
        "filename": file.filename,
        "chunks_indexed": total_added,
        "total_vector_chunks": collection.count(),
        "total_lexical_terms": len(LEXICAL_KEYWORD_INDEX)
    }

def classify_query_intent_and_complexity(question: str) -> str:
    """Evaluates question syntax and intent to select the optimal retrieval tier."""
    q_lower = question.lower()
    analytical_cues = ["compare", "contrast", "architecture", "tradeoff", "discrepancy", "evaluate", "synthesize", "why", "evolution"]
    lookup_cues = ["what is the error code", "section", "table", "port", "id", "status of", "define"]

    if any(cue in q_lower for cue in analytical_cues) or len(question.split()) > 15:
        return "DEEP_ANALYTIC"
    elif any(cue in q_lower for cue in lookup_cues) and len(question.split()) < 9:
        return "SIMPLE_LOOKUP"
    else:
        return "SEMANTIC_FACT"

def execute_tier1_lexical(query: str, target_doc: str = "ALL") -> Optional[str]:
    """Tier 1: High-speed exact lexical matching (<10ms)."""
    words = [w.lower() for w in query.split() if len(w) >= 4]
    candidates = []
    for w in words:
        if w in LEXICAL_KEYWORD_INDEX:
            candidates.extend(LEXICAL_KEYWORD_INDEX[w])

    if not candidates:
        return None

    filtered = [c for c in candidates if (target_doc == "ALL" or c["source"] == target_doc)]
    if not filtered:
        return None

    best = filtered[0]
    return f"[{best['source']} Page {best['page']}]: {best['chunk']}"

def execute_tier2_dense(query: str, target_doc: str = "ALL") -> str:
    """Tier 2: Standard dense vector retrieval (~40ms)."""
    if collection.count() == 0:
        return "No indexed vector documents available."

    q_vec = embed_model.encode(query).tolist()
    where_filter = {"source": target_doc} if target_doc != "ALL" else None
    results = collection.query(query_embeddings=[q_vec], n_results=min(2, collection.count()), where=where_filter)

    if not results['documents'] or not results['documents'][0]:
        return "No relevant passages found."

    m = results['metadatas'][0][0]
    return f"[{m.get('source')} Page {m.get('page')}]: {results['documents'][0][0]}"

def execute_tier3_deep_rerank(query: str, target_doc: str = "ALL") -> str:
    """Tier 3: Multi-chunk dense vector retrieval + Cross-Encoder re-ranking (~180ms)."""
    if collection.count() == 0:
        return "No indexed documents available."

    q_vec = embed_model.encode(query).tolist()
    where_filter = {"source": target_doc} if target_doc != "ALL" else None
    results = collection.query(query_embeddings=[q_vec], n_results=min(6, collection.count()), where=where_filter)

    if not results['documents'] or not results['documents'][0]:
        return "No relevant passages found."

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]

    blocks = []
    for idx in ranked_indices:
        m = metas[idx]
        blocks.append(f"[{m.get('source')} Page {m.get('page')} | Cross-Score: {scores[idx]:.2f}]:\n{candidates[idx]}")

    return "\n\n".join(blocks)

@app.post("/chat", response_model=AdaptiveRouterResponse)
def adaptive_router_chat(payload: AdaptiveRouterRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0 and len(LEXICAL_KEYWORD_INDEX) == 0:
        raise HTTPException(status_code=400, detail="Document repository is empty. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    start_total = time.time()

    # Step 1: Query Complexity Classification
    t_route_start = time.time()
    complexity = classify_query_intent_and_complexity(payload.question)
    routing_latency = round((time.time() - t_route_start) * 1000, 2)

    # Step 2: Adaptive Dispatching based on Complexity & Cost
    t_retrieval_start = time.time()
    routed_tier = "TIER_2_DENSE_VECTOR"
    retrieved_text = ""
    simulated_cost = 0.00010

    if complexity == "SIMPLE_LOOKUP":
        lex_hit = execute_tier1_lexical(payload.question, payload.target_doc)
        if lex_hit:
            retrieved_text = lex_hit
            routed_tier = "TIER_1_LEXICAL"
            simulated_cost = 0.00001
        else:
            # Fallback to dense if lexical index misses
            retrieved_text = execute_tier2_dense(payload.question, payload.target_doc)
            routed_tier = "TIER_2_DENSE_VECTOR"
            simulated_cost = 0.00010
    elif complexity == "DEEP_ANALYTIC":
        retrieved_text = execute_tier3_deep_rerank(payload.question, payload.target_doc)
        routed_tier = "TIER_3_RERANKED_DEEP"
        simulated_cost = 0.00055
    else:
        retrieved_text = execute_tier2_dense(payload.question, payload.target_doc)
        routed_tier = "TIER_2_DENSE_VECTOR"
        simulated_cost = 0.00010

    retrieval_latency = round((time.time() - t_retrieval_start) * 1000, 2)

    # Step 3: Response Synthesis
    t_synth_start = time.time()
    system_prompt = (
        f"You are an enterprise AI routed through {routed_tier} for {complexity} execution. "
        "Provide a factual, verified answer directly from the retrieved context."
    )
    user_msg = f"--- CONTEXT ---\n{retrieved_text}\n\n--- QUESTION ---\n{payload.question}"

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_msg})

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.2
        )
        final_answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    synth_latency = round((time.time() - t_synth_start) * 1000, 2)
    total_latency = round((time.time() - start_total) * 1000, 2)

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": final_answer})

    sla_met = total_latency <= payload.max_latency_budget_ms

    telemetry = RoutingTelemetry(
        detected_complexity=complexity,
        routed_tier=routed_tier,
        routing_latency_ms=routing_latency,
        retrieval_latency_ms=retrieval_latency,
        synthesis_latency_ms=synth_latency,
        total_pipeline_ms=total_latency,
        simulated_compute_cost_usd=simulated_cost,
        sla_met=sla_met
    )

    return AdaptiveRouterResponse(
        session_id=session_id,
        standalone_question=payload.question,
        telemetry=telemetry,
        retrieved_evidence_summary=retrieved_text,
        final_answer=final_answer
    )
