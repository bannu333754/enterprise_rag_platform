import os
import shutil
import chromadb
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(
    title="Day 38 RAPTOR Hierarchical Tree-Organized RAG API",
    version="1.0.0",
    description="Recursive hierarchical summarization, multi-tier chunk aggregation, and tree-structured retrieval"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day38_raptor_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day38"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class RaptorRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"
    target_tier: str = "ALL"  # "ALL", "TIER_0_LEAF", "TIER_1_CLUSTER", "TIER_2_MACRO"

class RaptorNodeHit(BaseModel):
    source: str
    tier: str
    node_id: str
    score: float
    content: str

class RaptorResponse(BaseModel):
    session_id: str
    standalone_question: str
    retrieved_nodes: List[RaptorNodeHit]
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 38 RAPTOR Hierarchical RAG",
        "total_nodes_in_tree": collection.count(),
        "active_sessions": len(SESSION_HISTORIES)
    }

def generate_summary(text_block: str, tier_level: str) -> str:
    """Generates an abstractive summary for higher tree tiers."""
    prompt = f"""You are a RAPTOR hierarchical summarization engine.
Condense the following content into a high-density, fact-preserving {tier_level} summary that captures overarching themes, key metrics, and architectural decisions.

Content:
{text_block}

Structured Summary:"""
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return f"Summary of {len(text_block.split())} words of text for {tier_level}."

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    reader = PdfReader(file_path)
    leaf_chunks = []
    
    # --- TIER 0: Leaf Granular Chunks Extraction ---
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        words = text.split()
        chunk_size = 75
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        for c_idx, chunk in enumerate(chunks):
            doc_id = f"{file.filename}_t0_p{page_idx + 1}_c{c_idx}"
            leaf_chunks.append({
                "id": doc_id,
                "text": chunk,
                "page": page_idx + 1,
                "source": file.filename
            })

    if not leaf_chunks:
        raise HTTPException(status_code=400, detail="No readable text found in PDF.")

    # Ingest Tier 0 (Leaves) into ChromaDB
    t0_docs = [c["text"] for c in leaf_chunks]
    t0_ids = [c["id"] for c in leaf_chunks]
    t0_vecs = embed_model.encode(t0_docs).tolist()
    t0_metas = [{"source": c["source"], "page": c["page"], "tier": "TIER_0_LEAF"} for c in leaf_chunks]

    collection.add(documents=t0_docs, embeddings=t0_vecs, metadatas=t0_metas, ids=t0_ids)

    # --- TIER 1: Sectional / Mid-Level Cluster Summaries ---
    tier1_summaries = []
    cluster_size = 4  # Aggregate every 4 leaves into one Tier-1 summary
    for i in range(0, len(leaf_chunks), cluster_size):
        cluster_leafs = leaf_chunks[i:i + cluster_size]
        combined_text = "\n\n".join([f"Passage (p.{c['page']}): {c['text']}" for c in cluster_leafs])
        t1_summary = generate_summary(combined_text, "Tier-1 Sectional")
        t1_id = f"{file.filename}_t1_cluster_{i // cluster_size}"
        
        pages_covered = f"{cluster_leafs[0]['page']}-{cluster_leafs[-1]['page']}"
        tier1_summaries.append({
            "id": t1_id,
            "text": t1_summary,
            "pages": pages_covered,
            "source": file.filename
        })

    t1_docs = [s["text"] for s in tier1_summaries]
    t1_ids = [s["id"] for s in tier1_summaries]
    t1_vecs = embed_model.encode(t1_docs).tolist()
    t1_metas = [{"source": s["source"], "page": 0, "tier": "TIER_1_CLUSTER", "pages_covered": s["pages"]} for s in tier1_summaries]

    collection.add(documents=t1_docs, embeddings=t1_vecs, metadatas=t1_metas, ids=t1_ids)

    # --- TIER 2: Root Macro-Level Summaries ---
    combined_t1_text = "\n\n".join([f"Section Summary ({s['pages']}): {s['text']}" for s in tier1_summaries])
    t2_macro_summary = generate_summary(combined_t1_text, "Tier-2 Document Macro")
    t2_id = f"{file.filename}_t2_macro_root"
    t2_vec = embed_model.encode(t2_macro_summary).tolist()

    collection.add(
        documents=[t2_macro_summary],
        embeddings=[t2_vec],
        metadatas=[{"source": file.filename, "page": 0, "tier": "TIER_2_MACRO", "pages_covered": "All"}],
        ids=[t2_id]
    )

    return {
        "status": "Success",
        "filename": file.filename,
        "tier_0_leaf_chunks": len(leaf_chunks),
        "tier_1_cluster_summaries": len(tier1_summaries),
        "tier_2_macro_roots": 1,
        "total_nodes_indexed": len(leaf_chunks) + len(tier1_summaries) + 1,
        "total_collection_nodes": collection.count()
    }

@app.post("/chat", response_model=RaptorResponse)
def raptor_chat(payload: RaptorRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="Tree empty. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    # Dynamic Tier-Constrained Retrieval
    and_filters = []
    if payload.target_doc != "ALL":
        and_filters.append({"source": {"$eq": payload.target_doc}})
    if payload.target_tier != "ALL":
        and_filters.append({"tier": {"$eq": payload.target_tier}})

    if len(and_filters) > 1:
        where_filter = {"$and": and_filters}
    elif len(and_filters) == 1:
        where_filter = and_filters[0]
    else:
        where_filter = None

    q_vec = embed_model.encode(payload.question).tolist()
    n_res = min(6, collection.count())

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=n_res,
        where=where_filter
    )

    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No matching nodes in RAPTOR tree.")

    candidates = results['documents'][0]
    metas = results['metadatas'][0]
    ids = results['ids'][0]

    # Cross-Encoder Re-ranking across tree layers
    pairs = [[payload.question, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    retrieved_nodes: List[RaptorNodeHit] = []
    context_blocks = []

    for idx in ranked_indices:
        m = metas[idx]
        tier = m.get("tier", "TIER_0_LEAF")
        sc = round(float(scores[idx]), 3)

        retrieved_nodes.append(RaptorNodeHit(
            source=m.get("source", "Unknown"),
            tier=tier,
            node_id=ids[idx],
            score=sc,
            content=candidates[idx]
        ))

        tier_badge = "🌳 [MACRO SUMMARY]" if tier == "TIER_2_MACRO" else ("🌿 [SECTION SUMMARY]" if tier == "TIER_1_CLUSTER" else "🍃 [LEAF CHUNK]")
        context_blocks.append(f"{tier_badge} (Node: {ids[idx]} | Source: {m.get('source')}):\n{candidates[idx]}")

    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an enterprise research assistant equipped with a hierarchical RAPTOR tree. "
        "Answer the user prompt by synthesizing both broad thematic summaries and granular factual leaf chunks."
    )
    user_msg = f"--- RAPTOR HIERARCHICAL TREE EVIDENCE ---\n{context_str}\n\n--- QUESTION ---\n{payload.question}"

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

    return RaptorResponse(
        session_id=session_id,
        standalone_question=payload.question,
        retrieved_nodes=retrieved_nodes,
        final_answer=final_answer
    )
