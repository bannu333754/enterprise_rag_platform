import os
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
    title="Day 45 Corrective Multi-Agent Swarm RAG API",
    version="1.0.0",
    description="Multi-agent corrective orchestration with confidence auditing and web fallback emulation"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day45_crag_swarm")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day45"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class CorrectiveRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"
    confidence_threshold: float = 0.40

class CorrectiveTelemetry(BaseModel):
    internal_confidence_score: float
    retrieval_verdict: str  # "CORRECT", "AMBIGUOUS", "INCORRECT"
    fallback_search_triggered: bool
    fallback_query: Optional[str] = None
    attribution_mode: str

class CorrectiveResponse(BaseModel):
    session_id: str
    standalone_question: str
    telemetry: CorrectiveTelemetry
    internal_evidence: str
    external_evidence: Optional[str] = None
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 45 Corrective Multi-Agent Swarm RAG",
        "total_chunks": collection.count(),
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
        "total_chunks": collection.count()
    }

def audit_retrieval_confidence(question: str, top_score: float, top_chunk: str) -> tuple[str, float]:
    """Audits whether internal retrieval is sufficient, ambiguous, or incorrect."""
    prompt = f"""You are an Autonomous Retrieval Confidence & Boundary Auditor.
Inspect the user question and the top retrieved chunk.

Question: "{question}"
Top Retrieved Chunk (Cross-Encoder score: {top_score}):
"{top_chunk}"

Determine retrieval quality:
- "CORRECT": Chunk directly answers the query with high relevance.
- "AMBIGUOUS": Chunk is tangentially related or incomplete.
- "INCORRECT": Chunk is irrelevant, missing required information, or unrelated.

Respond STRICTLY with JSON:
{{
  "verdict": "CORRECT" or "AMBIGUOUS" or "INCORRECT",
  "confidence_score": 0.0 to 1.0
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
        return data.get("verdict", "CORRECT"), float(data.get("confidence_score", 0.75))
    except Exception:
        verdict = "CORRECT" if top_score > 0.3 else "INCORRECT"
        return verdict, max(0.1, min(1.0, top_score + 0.5))

def execute_external_fallback_agent(question: str) -> tuple[str, str]:
    """Simulates targeted external search synthesis when local docs fail."""
    search_query = f"{question} verified technical specification"
    prompt = f"""You are an External Fallback Web Search Agent.
The internal knowledge base lacked sufficient documentation. Generate a factual, verified external summary answering this query.

Target Query: "{search_query}"
Factual External Summary (2-3 concise sentences):"""
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return search_query, res.choices[0].message.content.strip()
    except Exception:
        return search_query, "External fallback agent unable to resolve external reference."

@app.post("/chat", response_model=CorrectiveResponse)
def corrective_swarm_chat(payload: CorrectiveRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    internal_chunk = "No local documents indexed."
    top_score = -1.0
    ref = "None"

    # Step 1: Internal Dense Retrieval & Re-ranking
    if collection.count() > 0:
        q_vec = embed_model.encode(payload.question).tolist()
        where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None
        n_res = min(4, collection.count())

        results = collection.query(query_embeddings=[q_vec], n_results=n_res, where=where_filter)
        if results['documents'] and results['documents'][0]:
            candidates = results['documents'][0]
            metas = results['metadatas'][0]

            pairs = [[payload.question, doc] for doc in candidates]
            scores = reranker.predict(pairs)
            best_idx = int(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[0])

            top_score = float(scores[best_idx])
            internal_chunk = candidates[best_idx]
            m = metas[best_idx]
            ref = f"{m.get('source')} Page {m.get('page')}"

    # Step 2: Retrieval Confidence Auditing
    verdict, confidence = audit_retrieval_confidence(payload.question, top_score, internal_chunk)

    fallback_triggered = False
    external_evidence = None
    fallback_q = None
    attribution = "INTERNAL_ONLY"

    # Step 3: Dynamic Branching & Fallback Orchestration
    if verdict == "INCORRECT" or confidence < payload.confidence_threshold:
        fallback_triggered = True
        attribution = "EXTERNAL_FALLBACK"
        fallback_q, external_evidence = execute_external_fallback_agent(payload.question)
    elif verdict == "AMBIGUOUS":
        fallback_triggered = True
        attribution = "HYBRID_FUSION"
        fallback_q, external_evidence = execute_external_fallback_agent(payload.question)

    # Step 4: Corrective Synthesis Arbiter
    context_sections = []
    if attribution in ["INTERNAL_ONLY", "HYBRID_FUSION"]:
        context_sections.append(f"--- INTERNAL REPOSITORY EVIDENCE [{ref}] ---\n{internal_chunk}")
    if external_evidence:
        context_sections.append(f"--- EXTERNAL FALLBACK AGENT KNOWLEDGE ---\n{external_evidence}")

    combined_context = "\n\n".join(context_sections)

    system_prompt = (
        "You are an enterprise Corrective RAG Arbiter. Synthesize an authoritative answer. "
        "Strictly attribute facts to either Internal Documents or External Fallback Knowledge."
    )
    user_msg = f"{combined_context}\n\n--- QUESTION ---\n{payload.question}"

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

    telemetry = CorrectiveTelemetry(
        internal_confidence_score=round(confidence, 3),
        retrieval_verdict=verdict,
        fallback_search_triggered=fallback_triggered,
        fallback_query=fallback_q,
        attribution_mode=attribution
    )

    return CorrectiveResponse(
        session_id=session_id,
        standalone_question=payload.question,
        telemetry=telemetry,
        internal_evidence=f"[{ref}] {internal_chunk}",
        external_evidence=external_evidence,
        final_answer=final_answer
    )
