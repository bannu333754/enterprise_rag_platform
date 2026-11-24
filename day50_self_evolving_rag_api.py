import os
import json
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
    title="Day 50 Enterprise Autonomous Self-Evolving RAG API",
    version="1.0.0",
    description="Active learning, automated hard-negative mining, synthetic golden dataset generation, and dynamic hyperparameter optimization"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day50_self_evolving_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day50"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Persistent In-Memory System State & Self-Tuning Parameters
SYSTEM_CONFIG = {
    "top_k_retrieval": 4,
    "rerank_threshold": 0.25,
    "dense_weight": 0.70,
    "active_version": "v1.0.0-initial"
}

GOLDEN_DATASET: List[Dict[str, str]] = []
HARD_NEGATIVES_POOL: List[Dict[str, str]] = []
INTERACTION_LOGS: List[Dict[str, object]] = []
SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class EvolutionQueryRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class FeedbackRequest(BaseModel):
    query_id: str
    is_accurate: bool
    user_notes: Optional[str] = None

class TuningMetrics(BaseModel):
    golden_eval_samples: int
    mined_hard_negatives: int
    optimal_top_k: int
    optimal_rerank_threshold: float
    simulated_mrr_score: float
    applied_version: str

class EvolutionResponse(BaseModel):
    query_id: str
    session_id: str
    standalone_question: str
    active_config: Dict[str, object]
    retrieved_chunk: str
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 50 Self-Evolving Autonomous RAG",
        "total_chunks": collection.count(),
        "golden_samples": len(GOLDEN_DATASET),
        "hard_negatives_logged": len(HARD_NEGATIVES_POOL),
        "current_config": SYSTEM_CONFIG
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

            # Autonomously synthesize golden QA pairs every 3 chunks
            if c_idx % 3 == 0 and len(chunk) > 100:
                synthesize_golden_pair(chunk, file.filename, page_idx + 1)

    return {
        "status": "Success",
        "filename": file.filename,
        "chunks_indexed": total_added,
        "golden_dataset_size": len(GOLDEN_DATASET),
        "total_chunks": collection.count()
    }

def synthesize_golden_pair(context: str, source: str, page: int):
    """Generates synthetic questions and ground-truth answers for automated benchmarking."""
    prompt = f"""You are an Autonomous Golden Dataset Synthesizer.
From the provided text chunk, extract one high-value factual question and its definitive concise answer.

Context:
{context}

Respond STRICTLY in JSON:
{{
  "synthetic_question": "...",
  "ground_truth_answer": "..."
}}"""
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        raw = res.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        data = json.loads(raw.strip())
        GOLDEN_DATASET.append({
            "question": data.get("synthetic_question"),
            "ground_truth": data.get("ground_truth_answer"),
            "source": source,
            "page": page,
            "reference_chunk": context
        })
    except Exception:
        pass

@app.post("/chat", response_model=EvolutionResponse)
def evolving_chat(payload: EvolutionQueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="Repository is empty. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    query_id = f"q_{int(time.time() * 1000)}"

    # Retrieve using active dynamically tuned hyperparameter config
    q_vec = embed_model.encode(payload.question).tolist()
    where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None
    k_val = min(SYSTEM_CONFIG["top_k_retrieval"], collection.count())

    results = collection.query(query_embeddings=[q_vec], n_results=k_val, where=where_filter)
    candidates = results['documents'][0] if results['documents'] else []
    metas = results['metadatas'][0] if results['metadatas'] else []

    if not candidates:
        raise HTTPException(status_code=404, detail="No matching documents found.")

    pairs = [[payload.question, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    best_idx = ranked[0]
    top_score = float(scores[best_idx])
    best_chunk = candidates[best_idx]
    best_meta = metas[best_idx]

    # Automated Hard Negative Mining: High vector rank but low cross-encoder score
    if len(ranked) > 1:
        tail_idx = ranked[-1]
        tail_score = float(scores[tail_idx])
        if tail_score < SYSTEM_CONFIG["rerank_threshold"]:
            HARD_NEGATIVES_POOL.append({
                "query": payload.question,
                "negative_chunk": candidates[tail_idx],
                "cross_score": tail_score,
                "detected_at": time.strftime("%Y-%m-%d %H:%M:%S")
            })

    # Response Generation
    ref = f"{best_meta.get('source')} Page {best_meta.get('page')}"
    prompt = f"""You are an enterprise AI powered by a Self-Evolving RAG Engine.
Answer the user question strictly using the retrieved context.

Context [{ref}]:
{best_chunk}

Question: {payload.question}
Answer:"""

    messages = [{"role": "system", "content": prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": payload.question})

    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.2
        )
        final_answer = res.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": final_answer})

    INTERACTION_LOGS.append({
        "query_id": query_id,
        "question": payload.question,
        "retrieved_chunk": best_chunk,
        "top_score": top_score,
        "answer": final_answer,
        "config_version": SYSTEM_CONFIG["active_version"]
    })

    return EvolutionResponse(
        query_id=query_id,
        session_id=session_id,
        standalone_question=payload.question,
        active_config=SYSTEM_CONFIG.copy(),
        retrieved_chunk=f"[{ref} | Score: {top_score:.3f}]: {best_chunk}",
        final_answer=final_answer
    )

@app.post("/feedback")
def submit_feedback(payload: FeedbackRequest):
    """Logs human-in-the-loop active learning feedback."""
    for item in INTERACTION_LOGS:
        if item.get("query_id") == payload.query_id:
            item["user_feedback"] = "ACCURATE" if payload.is_accurate else "INACCURATE"
            item["user_notes"] = payload.user_notes
            if not payload.is_accurate:
                HARD_NEGATIVES_POOL.append({
                    "query": str(item.get("question")),
                    "negative_chunk": str(item.get("retrieved_chunk")),
                    "cross_score": float(item.get("top_score", 0.0)),
                    "detected_at": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            return {"status": "Feedback recorded", "query_id": payload.query_id}
    raise HTTPException(status_code=404, detail="Query ID not found in interaction logs.")

@app.post("/self-tune", response_model=TuningMetrics)
def trigger_autonomous_self_tuning():
    """Runs hyperparameter grid optimization against the synthetic golden dataset."""
    if not GOLDEN_DATASET:
        raise HTTPException(status_code=400, detail="Golden dataset is empty. Upload documents first to generate evaluation pairs.")

    candidate_top_ks = [2, 4, 6]
    candidate_thresholds = [0.15, 0.25, 0.35]

    best_mrr = 0.0
    best_k = SYSTEM_CONFIG["top_k_retrieval"]
    best_thresh = SYSTEM_CONFIG["rerank_threshold"]

    # Grid search evaluation over golden dataset
    for k in candidate_top_ks:
        for th in candidate_thresholds:
            reciprocal_ranks = []
            for item in GOLDEN_DATASET[:6]:
                q_vec = embed_model.encode(item["question"]).tolist()
                res = collection.query(query_embeddings=[q_vec], n_results=min(k, collection.count()))
                docs = res['documents'][0] if res['documents'] else []

                rank = 0
                for idx, doc in enumerate(docs):
                    if item["source"] in doc or item["ground_truth"][:30] in doc:
                        rank = idx + 1
                        break
                reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)

            avg_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
            if avg_mrr >= best_mrr:
                best_mrr = avg_mrr
                best_k = k
                best_thresh = th

    new_version = f"v{int(time.time())}-optimized"
    SYSTEM_CONFIG["top_k_retrieval"] = best_k
    SYSTEM_CONFIG["rerank_threshold"] = best_thresh
    SYSTEM_CONFIG["active_version"] = new_version

    return TuningMetrics(
        golden_eval_samples=len(GOLDEN_DATASET),
        mined_hard_negatives=len(HARD_NEGATIVES_POOL),
        optimal_top_k=best_k,
        optimal_rerank_threshold=best_thresh,
        simulated_mrr_score=round(best_mrr, 4),
        applied_version=new_version
    )
