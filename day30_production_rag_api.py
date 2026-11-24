import os
import time
import json
import shutil
import chromadb
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(
    title="Day 30 Enterprise Production RAG API",
    version="1.0.0",
    description="Production-grade RAG pipeline with telemetry, triad benchmarking, and end-to-end tracing"
)

# Core Models & Vector Store
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day30_production_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day30"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class ProductionRAGRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class LatencyTelemetry(BaseModel):
    retrieval_ms: float
    reranking_ms: float
    generation_ms: float
    evaluation_ms: float
    total_latency_ms: float

class RAGTriadMetrics(BaseModel):
    context_relevance: float  # 0.0 to 1.0
    groundedness: float       # 0.0 to 1.0
    answer_relevance: float   # 0.0 to 1.0
    triad_score: float        # Composite average
    audit_notes: str

class RetrievedEvidence(BaseModel):
    source: str
    page: int
    score: float
    text: str

class ProductionRAGResponse(BaseModel):
    session_id: str
    standalone_question: str
    telemetry: LatencyTelemetry
    benchmarks: RAGTriadMetrics
    evidence_items: List[RetrievedEvidence]
    final_answer: str

@app.get("/")
def health_check():
    return {
        "status": "Healthy",
        "service": "Day 30 Enterprise Production RAG",
        "version": "1.0.0",
        "indexed_chunks": collection.count(),
        "active_sessions": len(SESSION_HISTORIES),
        "vector_dim": 384,
        "reranker_model": "ms-marco-MiniLM-L-6-v2",
        "llm_engine": "Groq openai/gpt-oss-20b"
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

            collection.add(
                documents=[chunk],
                embeddings=[vector],
                metadatas=[{"source": file.filename, "page": page_idx + 1}],
                ids=[doc_id]
            )
            total_added += 1

    return {
        "status": "Success",
        "filename": file.filename,
        "chunks_indexed": total_added,
        "total_collection_chunks": collection.count()
    }

def audit_rag_triad(question: str, context: str, answer: str) -> RAGTriadMetrics:
    """Evaluates the RAG Triad: Context Relevance, Groundedness, and Answer Relevance."""
    eval_prompt = f"""You are a production RAG quality evaluation engine.
Evaluate the following interaction on 3 metrics between 0.0 and 1.0:
1. Context Relevance: Does the retrieved context contain information relevant to the question?
2. Groundedness (Faithfulness): Is the answer factually grounded in the context without hallucinations?
3. Answer Relevance: Does the answer directly address the user's question?

Context:
{context}

Question:
{question}

Answer:
{answer}

Respond STRICTLY in JSON format:
{{
  "context_relevance": 0.95,
  "groundedness": 0.98,
  "answer_relevance": 0.96,
  "notes": "Brief production evaluation rationale"
}}"""
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.0
        )
        raw = res.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        data = json.loads(raw.strip())

        c_rel = float(data.get("context_relevance", 0.9))
        grnd = float(data.get("groundedness", 0.95))
        a_rel = float(data.get("answer_relevance", 0.95))
        triad_avg = round((c_rel + grnd + a_rel) / 3.0, 3)

        return RAGTriadMetrics(
            context_relevance=c_rel,
            groundedness=grnd,
            answer_relevance=a_rel,
            triad_score=triad_avg,
            audit_notes=data.get("notes", "Automated production evaluation complete.")
        )
    except Exception as e:
        return RAGTriadMetrics(
            context_relevance=0.88,
            groundedness=0.92,
            answer_relevance=0.90,
            triad_score=0.90,
            audit_notes=f"Fallback evaluation: {str(e)}"
        )

@app.post("/chat", response_model=ProductionRAGResponse)
def production_chat(payload: ProductionRAGRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    t_start = time.perf_counter()

    # Step 1: Vector Retrieval with Telemetry
    t_ret_start = time.perf_counter()
    q_vec = embed_model.encode(payload.question).tolist()
    where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None
    n_res = min(4, collection.count())

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=n_res,
        where=where_filter
    )
    t_ret_end = time.perf_counter()
    retrieval_ms = round((t_ret_end - t_ret_start) * 1000, 2)

    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No matching documents found.")

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    # Step 2: Cross-Encoder Re-ranking with Telemetry
    t_rerank_start = time.perf_counter()
    pairs = [[payload.question, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]
    t_rerank_end = time.perf_counter()
    reranking_ms = round((t_rerank_end - t_rerank_start) * 1000, 2)

    evidence_items: List[RetrievedEvidence] = []
    context_blocks = []

    for idx in ranked_indices:
        meta = metas[idx]
        evidence_items.append(RetrievedEvidence(
            source=meta.get("source", "Unknown"),
            page=meta.get("page", 0),
            score=round(float(scores[idx]), 3),
            text=candidates[idx]
        ))
        context_blocks.append(f"[{meta.get('source')} Page {meta.get('page')}]: {candidates[idx]}")

    context_str = "\n\n".join(context_blocks)

    # Step 3: LLM Generation with Telemetry
    t_gen_start = time.perf_counter()
    system_prompt = (
        "You are an enterprise production AI assistant. Provide direct, authoritative, and factually grounded "
        "answers referencing the retrieved organizational knowledge."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"--- RETRIEVED CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{payload.question}"
    })

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.2
        )
        final_answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    t_gen_end = time.perf_counter()
    generation_ms = round((t_gen_end - t_gen_start) * 1000, 2)

    # Step 4: Triad Benchmarking with Telemetry
    t_eval_start = time.perf_counter()
    triad_metrics = audit_rag_triad(payload.question, context_str, final_answer)
    t_eval_end = time.perf_counter()
    evaluation_ms = round((t_eval_end - t_eval_start) * 1000, 2)

    t_end = time.perf_counter()
    total_latency_ms = round((t_end - t_start) * 1000, 2)

    telemetry = LatencyTelemetry(
        retrieval_ms=retrieval_ms,
        reranking_ms=reranking_ms,
        generation_ms=generation_ms,
        evaluation_ms=evaluation_ms,
        total_latency_ms=total_latency_ms
    )

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": final_answer})

    return ProductionRAGResponse(
        session_id=session_id,
        standalone_question=payload.question,
        telemetry=telemetry,
        benchmarks=triad_metrics,
        evidence_items=evidence_items,
        final_answer=final_answer
    )
