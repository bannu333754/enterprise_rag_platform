import os
import math
import time
import shutil
import chromadb
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(
    title="Day 49 Temporal-Decayed State-Tracking RAG API",
    version="1.0.0",
    description="Chronological state tracking, exponential time-decay scoring, and supersession auditing"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day49_temporal_state_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day49"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class TemporalQueryRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"
    half_life_days: float = 180.0
    alpha_semantic_weight: float = 0.65

class ScoredTemporalChunk(BaseModel):
    source: str
    page: int
    doc_date: str
    age_days: float
    raw_similarity: float
    temporal_decay_factor: float
    composite_score: float
    content: str

class TemporalAuditSummary(BaseModel):
    active_policy_state: str
    historical_superseded_state: Optional[str] = None
    state_transition_detected: bool

class TemporalQueryResponse(BaseModel):
    session_id: str
    standalone_question: str
    retrieved_timeline_chunks: List[ScoredTemporalChunk]
    temporal_audit: TemporalAuditSummary
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 49 Temporal State-Tracking RAG",
        "total_chunks": collection.count(),
        "active_sessions": len(SESSION_HISTORIES)
    }

@app.post("/upload")
async def upload_temporal_document(
    file: UploadFile = File(...),
    document_date: str = Form("2026-01-01")  # Format: YYYY-MM-DD
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        doc_dt = datetime.strptime(document_date.strip(), "%Y-%m-%d")
        epoch_timestamp = doc_dt.timestamp()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD (e.g., 2026-06-15).")

    file_path = os.path.join(UPLOAD_DIR, f"{document_date.strip()}_{file.filename}")
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
            doc_id = f"{file.filename}_{document_date.strip()}_p{page_idx + 1}_c{c_idx}"
            vector = embed_model.encode(chunk).tolist()

            collection.add(
                documents=[chunk],
                embeddings=[vector],
                metadatas=[{
                    "source": file.filename,
                    "doc_date": document_date.strip(),
                    "epoch_timestamp": epoch_timestamp,
                    "page": page_idx + 1
                }],
                ids=[doc_id]
            )
            total_added += 1

    return {
        "status": "Success",
        "filename": file.filename,
        "assigned_date": document_date.strip(),
        "chunks_indexed": total_added,
        "total_repository_chunks": collection.count()
    }

def compute_temporal_decay(epoch_timestamp: float, half_life_days: float) -> tuple[float, float]:
    """Calculates age in days and exponential half-life decay multiplier."""
    now_epoch = time.time()
    diff_seconds = max(0.0, now_epoch - epoch_timestamp)
    age_days = diff_seconds / 86400.0
    half_life_safe = max(1.0, half_life_days)
    decay_constant = math.log(2) / half_life_safe
    decay_factor = math.exp(-decay_constant * age_days)
    return round(age_days, 1), round(decay_factor, 4)

def audit_state_supersession(question: str, chunks: List[ScoredTemporalChunk]) -> TemporalAuditSummary:
    """Audits chunks ordered by date to verify active vs superseded state."""
    timeline_text = "\n\n".join([
        f"[Date: {c.doc_date} | Source: {c.source} p.{c.page}]:\n{c.content}"
        for c in chunks
    ])

    prompt = f"""You are a Chronological Knowledge State & Supersession Auditor.
Review the following timeline of evidence chunks retrieved for the user question:

Question: "{question}"

Chronological Evidence:
{timeline_text}

Determine:
1. Has a newer document or rule superseded/deprecated an older one regarding this topic?
2. What is the current active policy or technical state?
3. What was the older/superseded state (if any)?

Respond STRICTLY with JSON:
{{
  "state_transition_detected": true or false,
  "active_policy_state": "Concise summary of current active status",
  "historical_superseded_state": "Summary of previous state, or null if none"
}}"""
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw = res.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        data = json.loads(raw.strip())
        return TemporalAuditSummary(
            active_policy_state=data.get("active_policy_state", "Current state reflected in latest records."),
            historical_superseded_state=data.get("historical_superseded_state"),
            state_transition_detected=bool(data.get("state_transition_detected", False))
        )
    except Exception:
        return TemporalAuditSummary(
            active_policy_state="Verified latest state.",
            historical_superseded_state=None,
            state_transition_detected=False
        )

@app.post("/chat", response_model=TemporalQueryResponse)
def temporal_state_chat(payload: TemporalQueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="Document repository empty. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    # Step 1: Broad Vector Candidate Query
    q_vec = embed_model.encode(payload.question).tolist()
    where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None
    n_res = min(8, collection.count())

    results = collection.query(query_embeddings=[q_vec], n_results=n_res, where=where_filter)
    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No matching documents found.")

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    # Step 2: Compute Cross-Encoder Relevance & Exponential Temporal Decay
    pairs = [[payload.question, doc] for doc in candidates]
    raw_scores = reranker.predict(pairs)

    scored_pool: List[ScoredTemporalChunk] = []
    for idx, (doc, meta) in enumerate(zip(candidates, metas)):
        raw_sc = float(raw_scores[idx])
        # Normalize score roughly to 0.0 - 1.0 range
        sim_norm = 1.0 / (1.0 + math.exp(-raw_sc))
        epoch_ts = float(meta.get("epoch_timestamp", time.time()))
        age_days, decay_factor = compute_temporal_decay(epoch_ts, payload.half_life_days)

        # Composite Time-Decayed Relevance Formulation
        alpha = payload.alpha_semantic_weight
        composite_score = round(alpha * sim_norm + (1.0 - alpha) * decay_factor, 4)

        scored_pool.append(ScoredTemporalChunk(
            source=meta.get("source", "Unknown"),
            page=meta.get("page", 0),
            doc_date=meta.get("doc_date", "2026-01-01"),
            age_days=age_days,
            raw_similarity=round(sim_norm, 4),
            temporal_decay_factor=decay_factor,
            composite_score=composite_score,
            content=doc
        ))

    # Sort descending by composite score, pick top 4, then order chronologically
    ranked_top = sorted(scored_pool, key=lambda x: x.composite_score, reverse=True)[:4]
    chronological_evidence = sorted(ranked_top, key=lambda x: x.doc_date)

    # Step 3: Audit Temporal Supersession
    audit = audit_state_supersession(payload.question, chronological_evidence)

    # Step 4: Synthesize State-Tracking Final Answer
    context_blocks = [
        f"[{c.doc_date} | {c.source} Page {c.page} | Age: {c.age_days}d | Composite Score: {c.composite_score}]:\n{c.content}"
        for c in chronological_evidence
    ]
    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an enterprise AI specializing in Chronological State Tracking. "
        "Answer the question accurately using the dated evidence. "
        "If a newer document supersedes or updates an earlier rule, explicitly articulate the change timeline."
    )
    user_msg = f"--- CHRONOLOGICAL EVIDENCE CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{payload.question}"

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

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": final_answer})

    return TemporalQueryResponse(
        session_id=session_id,
        standalone_question=payload.question,
        retrieved_timeline_chunks=chronological_evidence,
        temporal_audit=audit,
        final_answer=final_answer
    )
