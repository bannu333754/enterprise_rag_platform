import os
import asyncio
import shutil
import chromadb
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(
    title="Day 31 Distributed Sharded Multi-Index RAG API",
    version="1.0.0",
    description="Multi-shard asynchronous retrieval with Reciprocal Rank Fusion (RRF)"
)

# Core Models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()

# Distinct Vector Shards
SHARDS = ["tech_engineering", "business_policy", "operations_general"]
shard_collections = {
    shard: chroma_client.get_or_create_collection(name=f"shard_{shard}")
    for shard in SHARDS
}

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day31"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class DistributedRequest(BaseModel):
    session_id: str
    question: str
    target_shards: List[str] = ["ALL"]

class ShardHit(BaseModel):
    shard: str
    source: str
    page: int
    rrf_score: float
    content: str

class DistributedResponse(BaseModel):
    session_id: str
    standalone_question: str
    active_shards_queried: List[str]
    merged_results: List[ShardHit]
    final_answer: str

@app.get("/")
def cluster_status():
    stats = {shard: shard_collections[shard].count() for shard in SHARDS}
    return {
        "status": "Online",
        "service": "Day 31 Distributed Sharded RAG Cluster",
        "available_shards": SHARDS,
        "shard_chunk_counts": stats,
        "total_cluster_chunks": sum(stats.values()),
        "active_sessions": len(SESSION_HISTORIES)
    }

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    assigned_shard: str = Form("operations_general")
):
    if assigned_shard not in SHARDS:
        raise HTTPException(status_code=400, detail=f"Invalid shard. Choose from: {SHARDS}")
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    reader = PdfReader(file_path)
    target_coll = shard_collections[assigned_shard]
    total_added = 0

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        words = text.split()
        chunk_size = 75
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        for c_idx, chunk in enumerate(chunks):
            doc_id = f"{assigned_shard}_{file.filename}_p{page_idx + 1}_c{c_idx}"
            vector = embed_model.encode(chunk).tolist()

            target_coll.add(
                documents=[chunk],
                embeddings=[vector],
                metadatas=[{"source": file.filename, "page": page_idx + 1, "shard": assigned_shard}],
                ids=[doc_id]
            )
            total_added += 1

    return {
        "status": "Success",
        "filename": file.filename,
        "assigned_shard": assigned_shard,
        "chunks_indexed": total_added,
        "shard_total_chunks": target_coll.count()
    }

async def async_search_shard(shard_name: str, query_vec: List[float], n_results: int = 3):
    """Asynchronous worker querying an individual vector shard."""
    loop = asyncio.get_event_loop()
    coll = shard_collections[shard_name]
    if coll.count() == 0:
        return shard_name, []

    res = await loop.run_in_executor(
        None,
        lambda: coll.query(query_embeddings=[query_vec], n_results=min(n_results, coll.count()))
    )

    items = []
    if res['documents'] and res['documents'][0]:
        docs = res['documents'][0]
        metas = res['metadatas'][0]
        for rank, (doc, meta) in enumerate(zip(docs, metas)):
            items.append({
                "shard": shard_name,
                "text": doc,
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", 0),
                "rank": rank + 1
            })
    return shard_name, items

def reciprocal_rank_fusion(shard_results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
    """Combines multi-shard search results using Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}

    for shard_items in shard_results_list:
        for item in shard_items:
            key = item["text"]
            if key not in scores:
                scores[key] = 0.0
                doc_map[key] = item
            scores[key] += 1.0 / (k + item["rank"])

    # Sort merged documents by composite RRF score
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused_list = []
    for key in sorted_keys:
        merged_item = doc_map[key].copy()
        merged_item["rrf_score"] = round(scores[key], 4)
        fused_list.append(merged_item)
    return fused_list

@app.post("/chat", response_model=DistributedResponse)
async def distributed_chat(payload: DistributedRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    total_cluster_count = sum(coll.count() for coll in shard_collections.values())
    if total_cluster_count == 0:
        raise HTTPException(status_code=400, detail="Cluster empty. Upload documents into shards first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    active_shards = SHARDS if "ALL" in payload.target_shards else [s for s in payload.target_shards if s in SHARDS]
    if not active_shards:
        active_shards = SHARDS

    q_vec = embed_model.encode(payload.question).tolist()

    # Map Phase: Parallel Asynchronous Dispatch across Shards
    tasks = [async_search_shard(shard, q_vec, n_results=3) for shard in active_shards]
    shard_outputs = await asyncio.gather(*tasks)

    all_shard_items = [items for _, items in shard_outputs]

    # Reduce Phase: Reciprocal Rank Fusion
    fused_candidates = reciprocal_rank_fusion(all_shard_items, k=60)
    if not fused_candidates:
        raise HTTPException(status_code=404, detail="No matching documents found across queried shards.")

    # Cross-Encoder Fine-Grained Re-ranking on Fused Top Results
    top_fused = fused_candidates[:5]
    pairs = [[payload.question, c["text"]] for c in top_fused]
    ce_scores = reranker.predict(pairs)
    ranked_indices = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)[:3]

    merged_results: List[ShardHit] = []
    context_blocks = []

    for idx in ranked_indices:
        item = top_fused[idx]
        merged_results.append(ShardHit(
            shard=item["shard"],
            source=item["source"],
            page=item["page"],
            rrf_score=item["rrf_score"],
            content=item["text"]
        ))
        context_blocks.append(f"[{item['shard']} | {item['source']} Page {item['page']}]:\n{item['text']}")

    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an enterprise distributed intelligence assistant. Answer the question comprehensively, "
        "synthesizing evidence gathered across distributed departmental shards."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"--- SHARD CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{payload.question}"
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

    return DistributedResponse(
        session_id=session_id,
        standalone_question=payload.question,
        active_shards_queried=active_shards,
        merged_results=merged_results,
        final_answer=final_answer
    )
