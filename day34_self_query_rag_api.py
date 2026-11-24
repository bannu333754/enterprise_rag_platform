import os
import json
import shutil
import chromadb
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(
    title="Day 34 Self-Querying & Metadata Filter RAG API",
    version="1.0.0",
    description="LLM Query-to-Filter decomposition with dynamic ChromaDB AST translation"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day34_self_query_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day34"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class SelfQueryRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class ExtractedFilterPlan(BaseModel):
    semantic_query: str
    detected_category: Optional[str] = None
    min_year: Optional[int] = None
    target_department: Optional[str] = None
    raw_filter_json: str

class FilteredHit(BaseModel):
    source: str
    page: int
    category: str
    department: str
    year: int
    score: float
    content: str

class SelfQueryResponse(BaseModel):
    session_id: str
    standalone_question: str
    decomposition_plan: ExtractedFilterPlan
    retrieved_items: List[FilteredHit]
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 34 Self-Querying Metadata RAG",
        "total_chunks": collection.count(),
        "active_sessions": len(SESSION_HISTORIES)
    }

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    category: str = Form("Technical"),
    department: str = Form("Engineering"),
    pub_year: int = Form(2025)
):
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
                metadatas=[{
                    "source": file.filename,
                    "page": page_idx + 1,
                    "category": category,
                    "department": department,
                    "year": pub_year
                }],
                ids=[doc_id]
            )
            total_added += 1

    return {
        "status": "Success",
        "filename": file.filename,
        "metadata_assigned": {
            "category": category,
            "department": department,
            "year": pub_year
        },
        "chunks_indexed": total_added,
        "total_chunks": collection.count()
    }

def decompose_query_to_ast(user_query: str) -> Dict[str, Any]:
    """Decomposes a user query into a purely semantic search string and structured metadata filters."""
    prompt = f"""You are a Self-Querying AST Query Decomposer for a Vector Database.
Extract the pure semantic intent and any metadata attributes explicitly or implicitly mentioned.

Supported Schema Attributes:
- category: ["Technical", "Business", "Policy", "Compliance"]
- department: ["Engineering", "Finance", "Legal", "Operations"]
- year: Integer year (e.g., 2024, 2025, 2026)

User Query: "{user_query}"

Respond STRICTLY with a valid JSON object matching this structure:
{{
  "semantic_query": "search query stripped of metadata filters",
  "category": "detected category or null",
  "department": "detected department or null",
  "year": detected integer or null
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
        return json.loads(raw.strip())
    except Exception:
        return {
            "semantic_query": user_query,
            "category": None,
            "department": None,
            "year": None
        }

@app.post("/chat", response_model=SelfQueryResponse)
def self_query_chat(payload: SelfQueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    # Step 1: Query Decomposition into Semantic Intent + Metadata Filters
    plan_dict = decompose_query_to_ast(payload.question)
    semantic_str = plan_dict.get("semantic_query") or payload.question

    # Step 2: Build Native ChromaDB Metadata Where-Clause
    and_filters = []
    if payload.target_doc != "ALL":
        and_filters.append({"source": {"$eq": payload.target_doc}})
    if plan_dict.get("category"):
        and_filters.append({"category": {"$eq": plan_dict["category"]}})
    if plan_dict.get("department"):
        and_filters.append({"department": {"$eq": plan_dict["department"]}})
    if plan_dict.get("year"):
        and_filters.append({"year": {"$eq": int(plan_dict["year"])}})

    if len(and_filters) > 1:
        where_filter = {"$and": and_filters}
    elif len(and_filters) == 1:
        where_filter = and_filters[0]
    else:
        where_filter = None

    q_vec = embed_model.encode(semantic_str).tolist()
    n_res = min(6, collection.count())

    try:
        results = collection.query(
            query_embeddings=[q_vec],
            n_results=n_res,
            where=where_filter
        )
    except Exception:
        # Fallback to unrestricted search if filter conditions return no candidates
        results = collection.query(
            query_embeddings=[q_vec],
            n_results=n_res
        )

    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No documents matched the decomposed query filters.")

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    # Step 3: Cross-Encoder Re-ranking
    pairs = [[semantic_str, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]

    retrieved_items: List[FilteredHit] = []
    context_blocks = []

    for idx in ranked_idx:
        m = metas[idx]
        sc = round(float(scores[idx]), 3)
        retrieved_items.append(FilteredHit(
            source=m.get("source", "Unknown"),
            page=m.get("page", 0),
            category=m.get("category", "N/A"),
            department=m.get("department", "N/A"),
            year=m.get("year", 0),
            score=sc,
            content=candidates[idx]
        ))
        context_blocks.append(f"[{m.get('source')} | Dept: {m.get('department')} | Cat: {m.get('category')} | Year: {m.get('year')}]:\n{candidates[idx]}")

    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an enterprise AI with structured metadata self-querying capability. "
        "Answer the question directly, referencing facts from documents verified by the structured metadata filters."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"--- CONTEXT VERIFIED BY METADATA FILTERS ---\n{context_str}\n\n--- QUESTION ---\n{payload.question}"
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

    decomposition_plan = ExtractedFilterPlan(
        semantic_query=semantic_str,
        detected_category=plan_dict.get("category"),
        min_year=plan_dict.get("year"),
        target_department=plan_dict.get("department"),
        raw_filter_json=json.dumps(where_filter) if where_filter else "{}"
    )

    return SelfQueryResponse(
        session_id=session_id,
        standalone_question=payload.question,
        decomposition_plan=decomposition_plan,
        retrieved_items=retrieved_items,
        final_answer=final_answer
    )
