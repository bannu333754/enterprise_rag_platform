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
    title="Day 44 Self-RAG with Reflection Tokens API",
    version="1.0.0",
    description="Dynamic retrieval gating, relevance critique, grounding verification, and utility scoring"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day44_self_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day44"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class SelfRAGRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class PassageCritique(BaseModel):
    source: str
    page: int
    is_rel_token: str  # "[Relevant]" or "[Irrelevant]"
    relevance_rationale: str
    text_snippet: str

class SelfRAGTelemetry(BaseModel):
    retrieve_decision_token: str  # "[Retrieve]" or "[No-Retrieve]"
    total_candidates_inspected: int
    relevant_passages_accepted: int
    is_sup_token: str  # "[Fully-Supported]", "[Partially-Supported]", "[Not-Supported]"
    is_use_score: float  # 1.0 to 5.0

class SelfRAGResponse(BaseModel):
    session_id: str
    standalone_question: str
    telemetry: SelfRAGTelemetry
    passage_critiques: List[PassageCritique]
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 44 Self-RAG Reflection & Critique",
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

def gate_retrieval_decision(question: str) -> str:
    """Decides if the query requires external knowledge retrieval."""
    prompt = f"""You are a Self-RAG Retrieval Gater.
Determine if answering the following question requires retrieving specific external enterprise documentation, 
or if it is a general conversational/definitional query that can be answered directly without retrieval.

Question: "{question}"

Respond STRICTLY with JSON:
{{"decision": "[Retrieve]" or "[No-Retrieve]"}}"""
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
        return data.get("decision", "[Retrieve]")
    except Exception:
        return "[Retrieve]"

def critique_passages(question: str, candidates: List[str], metas: List[dict]) -> List[PassageCritique]:
    """Applies [IsRel] critique token to filter out irrelevant or distractor chunks."""
    critiques: List[PassageCritique] = []
    for doc, meta in zip(candidates, metas):
        prompt = f"""Critique this candidate passage for the question.
Question: "{question}"
Passage: "{doc}"

Determine relevance:
Respond STRICTLY with JSON:
{{
  "token": "[Relevant]" or "[Irrelevant]",
  "reason": "Short 1-sentence rationale"
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
            tok = data.get("token", "[Relevant]")
            reason = data.get("reason", "Evaluated against query intent.")
        except Exception:
            tok = "[Relevant]"
            reason = "Default fallback pass."

        critiques.append(PassageCritique(
            source=meta.get("source", "Unknown"),
            page=meta.get("page", 0),
            is_rel_token=tok,
            relevance_rationale=reason,
            text_snippet=doc
        ))
    return critiques

def critique_answer(question: str, context: str, answer: str) -> tuple[str, float]:
    """Computes [IsSup] (grounding) and [IsUse] (utility score)."""
    prompt = f"""You are a Self-RAG Output Critic.
Inspect the candidate answer against the retrieved context:

Context:
{context}

Question: "{question}"
Answer: "{answer}"

Evaluate:
1. Is the answer fully supported by context? ([Fully-Supported], [Partially-Supported], or [Not-Supported])
2. Utility score (1.0 to 5.0) for clarity, completeness, and directness.

Respond STRICTLY with JSON:
{{
  "is_sup": "[Fully-Supported]" or "[Partially-Supported]" or "[Not-Supported]",
  "is_use": 4.5
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
        return data.get("is_sup", "[Fully-Supported]"), float(data.get("is_use", 4.5))
    except Exception:
        return "[Fully-Supported]", 4.5

@app.post("/chat", response_model=SelfRAGResponse)
def self_rag_chat(payload: SelfRAGRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    # Step 1: [Retrieve] Gating
    retrieve_decision = gate_retrieval_decision(payload.question)

    accepted_passages = []
    critique_list: List[PassageCritique] = []

    if retrieve_decision == "[Retrieve]" and collection.count() > 0:
        q_vec = embed_model.encode(payload.question).tolist()
        where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None
        n_res = min(4, collection.count())

        results = collection.query(query_embeddings=[q_vec], n_results=n_res, where=where_filter)
        if results['documents'] and results['documents'][0]:
            candidates = results['documents'][0]
            metas = results['metadatas'][0]

            # Step 2: [IsRel] Passage Relevance Critiques
            critique_list = critique_passages(payload.question, candidates, metas)
            for c in critique_list:
                if c.is_rel_token == "[Relevant]":
                    accepted_passages.append(f"[{c.source} p.{c.page}]: {c.text_snippet}")

    context_str = "\n\n".join(accepted_passages) if accepted_passages else "No external retrieved context used."

    # Step 3: Response Synthesis
    system_prompt = (
        "You are an enterprise AI powered by Self-RAG. Synthesize an accurate, factual response. "
        "Rely strictly on provided context if available, otherwise answer clearly based on core general knowledge."
    )
    user_msg = f"--- CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{payload.question}"

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

    # Step 4: [IsSup] Grounding & [IsUse] Utility Critique
    is_sup, is_use = critique_answer(payload.question, context_str, final_answer)

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": final_answer})

    telemetry = SelfRAGTelemetry(
        retrieve_decision_token=retrieve_decision,
        total_candidates_inspected=len(critique_list),
        relevant_passages_accepted=len(accepted_passages),
        is_sup_token=is_sup,
        is_use_score=is_use
    )

    return SelfRAGResponse(
        session_id=session_id,
        standalone_question=payload.question,
        telemetry=telemetry,
        passage_critiques=critique_list,
        final_answer=final_answer
    )
