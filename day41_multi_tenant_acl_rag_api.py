import os
import json
import shutil
import chromadb
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(
    title="Day 41 Federated Multi-Tenant Enterprise RAG API",
    version="1.0.0",
    description="Multi-tenant isolation with fine-grained Document Access Control Lists (ACL)"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day41_enterprise_multi_tenant_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day41"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

# Security Hierarchy Definition
CLEARANCE_LEVELS = {
    "PUBLIC": 1,
    "INTERNAL": 2,
    "CONFIDENTIAL": 3,
    "RESTRICTED": 4
}

class TenantQueryRequest(BaseModel):
    session_id: str
    tenant_id: str
    user_id: str
    clearance_role: str  # "PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"
    question: str
    target_doc: str = "ALL"

class SecuredChunkHit(BaseModel):
    source: str
    page: int
    tenant_id: str
    required_clearance: str
    score: float
    text: str

class TenantQueryResponse(BaseModel):
    session_id: str
    tenant_id: str
    user_clearance: str
    applied_acl_filter: str
    accessible_evidence: List[SecuredChunkHit]
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 41 Multi-Tenant ACL Enterprise RAG",
        "total_secured_chunks": collection.count(),
        "active_sessions": len(SESSION_HISTORIES)
    }

@app.post("/upload")
async def upload_secured_document(
    file: UploadFile = File(...),
    tenant_id: str = Form(...),
    clearance_tier: str = Form("INTERNAL")
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    tier_clean = clearance_tier.upper().strip()
    if tier_clean not in CLEARANCE_LEVELS:
        raise HTTPException(status_code=400, detail=f"Invalid clearance tier. Must be one of {list(CLEARANCE_LEVELS.keys())}")

    tenant_clean = tenant_id.strip()
    if not tenant_clean:
        raise HTTPException(status_code=400, detail="Tenant ID cannot be empty.")

    file_path = os.path.join(UPLOAD_DIR, f"{tenant_clean}_{file.filename}")
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
            doc_id = f"{tenant_clean}_{file.filename}_p{page_idx + 1}_c{c_idx}"
            vector = embed_model.encode(chunk).tolist()

            collection.add(
                documents=[chunk],
                embeddings=[vector],
                metadatas=[{
                    "source": file.filename,
                    "tenant_id": tenant_clean,
                    "clearance": tier_clean,
                    "clearance_rank": CLEARANCE_LEVELS[tier_clean],
                    "page": page_idx + 1
                }],
                ids=[doc_id]
            )
            total_added += 1

    return {
        "status": "Success",
        "filename": file.filename,
        "tenant_id": tenant_clean,
        "clearance_assigned": tier_clean,
        "chunks_indexed": total_added,
        "total_chunks_in_index": collection.count()
    }

@app.post("/chat", response_model=TenantQueryResponse)
def tenant_secured_chat(payload: TenantQueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="Zero indexed records found. Upload documents first.")

    user_tier = payload.clearance_role.upper().strip()
    if user_tier not in CLEARANCE_LEVELS:
        raise HTTPException(status_code=403, detail="Invalid user security clearance credentials.")

    user_max_rank = CLEARANCE_LEVELS[user_tier]
    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    # Step 1: Construct Mandatory Tenant & ACL Predicates
    # Filter 1: Strict tenant isolation (cannot query other tenants' documents)
    # Filter 2: Clearance boundary (user rank >= document clearance rank)
    and_conditions = [
        {"tenant_id": {"$eq": payload.tenant_id}},
        {"clearance_rank": {"$lte": user_max_rank}}
    ]

    if payload.target_doc != "ALL":
        and_conditions.append({"source": {"$eq": payload.target_doc}})

    where_filter = {"$and": and_conditions}

    # Step 2: Vector Retrieval under Hard ACL Filter
    q_vec = embed_model.encode(payload.question).tolist()
    n_res = min(6, collection.count())

    try:
        results = collection.query(
            query_embeddings=[q_vec],
            n_results=n_res,
            where=where_filter
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ChromaDB filter exception: {str(e)}")

    if not results['documents'] or not results['documents'][0]:
        return TenantQueryResponse(
            session_id=session_id,
            tenant_id=payload.tenant_id,
            user_clearance=user_tier,
            applied_acl_filter=json.dumps(where_filter),
            accessible_evidence=[],
            final_answer="Access Denied or No Matching Evidence: You either lack the required clearance level to view documents matching this query, or no records exist within your tenant boundary."
        )

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    # Step 3: Re-rank Permitted Candidate Passages
    pairs = [[payload.question, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]

    evidence_hits: List[SecuredChunkHit] = []
    context_blocks = []

    for idx in ranked_indices:
        m = metas[idx]
        sc = round(float(scores[idx]), 3)
        doc_text = candidates[idx]

        evidence_hits.append(SecuredChunkHit(
            source=m.get("source", "Unknown"),
            page=m.get("page", 0),
            tenant_id=m.get("tenant_id", "Unknown"),
            required_clearance=m.get("clearance", "RESTRICTED"),
            score=sc,
            text=doc_text
        ))

        context_blocks.append(
            f"[{m.get('source')} | Clearance: {m.get('clearance')} | Page {m.get('page')}]:\n{doc_text}"
        )

    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an enterprise AI operating within an authenticated federated tenant environment. "
        "Answer the query factually based exclusively on the provided authorized evidence chunks. "
        "Never speculate beyond the permitted security boundary."
    )
    user_msg = f"--- AUTHORIZED CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{payload.question}"

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

    return TenantQueryResponse(
        session_id=session_id,
        tenant_id=payload.tenant_id,
        user_clearance=user_tier,
        applied_acl_filter=json.dumps(where_filter),
        accessible_evidence=evidence_hits,
        final_answer=final_answer
    )
