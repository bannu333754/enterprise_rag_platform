import os
import math
import shutil
import datetime
import chromadb
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(
    title="Day 33 Temporal & Time-Decayed RAG API",
    version="1.0.0",
    description="Time-decay vector scoring, chronological filtering, and recency-weighted retrieval"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day33_temporal_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day33"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class TemporalRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"
    decay_lambda: float = 0.005  # Decay coefficient per day
    max_age_days: Optional[int] = None  # Hard cutoff filter

class TemporalHit(BaseModel):
    source: str
    doc_date: str
    age_days: int
    raw_rerank_score: float
    time_decay_factor: float
    temporal_final_score: float
    content: str

class TemporalResponse(BaseModel):
    session_id: str
    standalone_question: str
    retrieved_items: List[TemporalHit]
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 33 Temporal Time-Decayed RAG",
        "total_chunks": collection.count(),
        "active_sessions": len(SESSION_HISTORIES)
    }

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    doc_date: str = Form(...)  # Expected format: YYYY-MM-DD
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        parsed_date = datetime.datetime.strptime(doc_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    reader = PdfReader(file_path)
    total_added = 0
    today = datetime.date.today()

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
                metadatas=[{
                    "source": file.filename,
                    "page": page_idx + 1,
                    "doc_date": parsed_date.strftime("%Y-%m-%d"),
                    "timestamp": parsed_date.toordinal()
                }],
                ids=[doc_id]
            )
            total_added += 1

    return {
        "status": "Success",
        "filename": file.filename,
        "assigned_date": parsed_date.strftime("%Y-%m-%d"),
        "chunks_indexed": total_added,
        "total_chunks": collection.count()
    }

@app.post("/chat", response_model=TemporalResponse)
def temporal_chat(payload: TemporalRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    q_vec = embed_model.encode(payload.question).tolist()
    where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None
    n_res = min(6, collection.count())

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=n_res,
        where=where_filter
    )

    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No matching documents found.")

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    # Step 1: Semantic Re-ranking via Cross-Encoder
    pairs = [[payload.question, doc] for doc in candidates]
    raw_scores = reranker.predict(pairs)

    today = datetime.date.today()
    scored_items = []

    # Step 2: Apply Exponential Time Decay & Optional Hard Cutoff
    for idx in range(len(candidates)):
        meta = metas[idx]
        doc_date_str = meta.get("doc_date", today.strftime("%Y-%m-%d"))
        doc_date = datetime.datetime.strptime(doc_date_str, "%Y-%m-%d").date()
        age_days = max(0, (today - doc_date).days)

        # Hard Cutoff Filter
        if payload.max_age_days is not None and age_days > payload.max_age_days:
            continue

        raw_score = float(raw_scores[idx])
        # Normalized logistic score from raw logit
        norm_semantic = 1.0 / (1.0 + math.exp(-raw_score))
        
        # Exponential decay factor: e^(-lambda * delta_days)
        decay_factor = math.exp(-payload.decay_lambda * age_days)
        temporal_score = norm_semantic * decay_factor

        scored_items.append({
            "source": meta.get("source", "Unknown"),
            "doc_date": doc_date_str,
            "age_days": age_days,
            "raw_score": round(raw_score, 3),
            "decay_factor": round(decay_factor, 4),
            "temporal_score": round(temporal_score, 4),
            "text": candidates[idx]
        })

    if not scored_items:
        raise HTTPException(
            status_code=404,
            detail="All candidate documents were filtered out by the max_age_days threshold."
        )

    # Rank by final time-decayed score
    scored_items.sort(key=lambda x: x["temporal_score"], reverse=True)
    top_items = scored_items[:2]

    retrieved_hits: List[TemporalHit] = []
    context_blocks = []

    for it in top_items:
        retrieved_hits.append(TemporalHit(
            source=it["source"],
            doc_date=it["doc_date"],
            age_days=it["age_days"],
            raw_rerank_score=it["raw_score"],
            time_decay_factor=it["decay_factor"],
            temporal_final_score=it["temporal_score"],
            content=it["text"]
        ))
        context_blocks.append(f"[{it['source']} | Date: {it['doc_date']} ({it['age_days']} days ago)]:\n{it['text']}")

    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an enterprise research assistant specializing in temporal knowledge retrieval. "
        "Answer the question accurately. When documents have different dates, prioritize newer revisions "
        "and mention dates explicitly when relevant."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"--- RETRIEVED CONTEXT WITH DATES ---\n{context_str}\n\n--- QUESTION ---\n{payload.question}"
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

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": final_answer})

    return TemporalResponse(
        session_id=session_id,
        standalone_question=payload.question,
        retrieved_items=retrieved_hits,
        final_answer=final_answer
    )
