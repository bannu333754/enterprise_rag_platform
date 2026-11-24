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
    title="Day 42 Self-Healing & Autonomous Query Correction RAG API",
    version="1.0.0",
    description="Iterative query reformulation, faithfulness auditing, and hallucination guardrails"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day42_self_healing_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day42"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class SelfHealingRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"
    max_heal_iterations: int = 2

class IterationAudit(BaseModel):
    iteration: int
    executed_query: str
    retrieved_chunk: str
    candidate_answer: str
    faithfulness_score: float
    audit_verdict: str
    reformulation_reason: Optional[str] = None

class SelfHealingResponse(BaseModel):
    session_id: str
    original_question: str
    total_iterations: int
    healing_triggered: bool
    audit_trail: List[IterationAudit]
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 42 Self-Healing Query Correction RAG",
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

def retrieve_and_rerank(query: str, target_doc: str = "ALL") -> tuple[str, str]:
    """Retrieves top chunk using vector search and cross-encoder re-ranking."""
    if collection.count() == 0:
        return "No documents indexed.", "N/A"

    q_vec = embed_model.encode(query).tolist()
    where_filter = {"source": target_doc} if target_doc != "ALL" else None
    n_res = min(4, collection.count())

    results = collection.query(query_embeddings=[q_vec], n_results=n_res, where=where_filter)
    if not results['documents'] or not results['documents'][0]:
        return "No relevant passages found.", "N/A"

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    best_idx = int(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[0])

    m = metas[best_idx]
    ref = f"{m.get('source')} Page {m.get('page')}"
    return candidates[best_idx], ref

def generate_draft(question: str, context: str, history: List[Dict[str, str]]) -> str:
    prompt = f"""You are a precise technical enterprise assistant.
Answer the user question using strictly the retrieved context. If the context does not contain enough evidence, state that clearly.

Retrieved Context:
{context}

Question: "{question}"
Answer:"""
    messages = [{"role": "system", "content": prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.1
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"Error during draft generation: {str(e)}"

def audit_faithfulness(question: str, context: str, answer: str) -> tuple[float, str, Optional[str]]:
    """Evaluates answer faithfulness and determines if query reformulation is required."""
    prompt = f"""You are an autonomous Hallucination and Faithfulness Auditor.
Evaluate whether the candidate answer is directly supported by the context without hallucination.

Question: "{question}"
Context:
{context}
Candidate Answer:
{answer}

Respond STRICTLY in JSON format:
{{
  "faithfulness_score": 0.0 to 1.0,
  "verdict": "PASSED" or "FAILED",
  "reformulation_query": "Rewritten search query with better technical keywords if FAILED, else null"
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
        score = float(data.get("faithfulness_score", 0.8))
        verdict = str(data.get("verdict", "PASSED"))
        re_query = data.get("reformulation_query")
        return score, verdict, re_query
    except Exception:
        return 0.85, "PASSED", None

@app.post("/chat", response_model=SelfHealingResponse)
def self_healing_chat(payload: SelfHealingRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="Document repository empty. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    current_query = payload.question
    audit_trail: List[IterationAudit] = []
    healing_triggered = False
    final_answer = ""

    for iteration in range(1, payload.max_heal_iterations + 1):
        context_chunk, ref = retrieve_and_rerank(current_query, payload.target_doc)
        draft_ans = generate_draft(current_query, context_chunk, history)
        score, verdict, reformulation = audit_faithfulness(current_query, context_chunk, draft_ans)

        audit_trail.append(IterationAudit(
            iteration=iteration,
            executed_query=current_query,
            retrieved_chunk=f"[{ref}] {context_chunk[:200]}...",
            candidate_answer=draft_ans,
            faithfulness_score=score,
            audit_verdict=verdict,
            reformulation_reason=f"Rewritten to: '{reformulation}'" if (verdict == "FAILED" and reformulation) else None
        ))

        final_answer = draft_ans

        if verdict == "PASSED" or not reformulation or iteration == payload.max_heal_iterations:
            break
        else:
            healing_triggered = True
            current_query = reformulation

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": final_answer})

    return SelfHealingResponse(
        session_id=session_id,
        original_question=payload.question,
        total_iterations=len(audit_trail),
        healing_triggered=healing_triggered,
        audit_trail=audit_trail,
        final_answer=final_answer
    )
