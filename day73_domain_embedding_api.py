import os
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [DOMAIN_EMBED] %(message)s")
logger = logging.getLogger("DomainEmbedAPI")

app = FastAPI(
    title="Day 73 Enterprise Fine-Tuned Domain Embedding Service",
    version="16.0.0",
    description="Vector search powered by domain-adapted embedding weights"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "./models/day73_enterprise_embedder"
try:
    if os.path.exists(MODEL_PATH) and os.listdir(MODEL_PATH):
        logger.info(f"Loading custom fine-tuned weights from {MODEL_PATH}")
        embed_model = SentenceTransformer(MODEL_PATH)
    else:
        logger.warning("Custom weights folder empty/not found; using base model.")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    logger.warning(f"Failed to load fine-tuned weights ({e}), falling back to base model.")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day73_proprietary_kb")
groq_client = Groq(api_key="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")

KB_ITEMS = [
    {
        "id": "doc_hyperion_db",
        "title": "HyperionDB Consensus Spec",
        "text": "HyperionDB employs a decentralized Raft variant called HyperRaft operating across multi-cloud regions with sub-5ms write acknowledgment."
    },
    {
        "id": "doc_chronos_queue",
        "title": "ChronosQueue Throughput Benchmarks",
        "text": "ChronosQueue partitions achieve up to 45,000 events/second using zero-copy memory mapping on dedicated NVMe drives."
    },
    {
        "id": "doc_aether_mesh",
        "title": "AetherMesh eBPF Security Spec",
        "text": "AetherMesh service mesh deploys dynamic eBPF socket filters to restrict cross-namespace pod traffic without sidecar proxies."
    }
]

def seed_database():
    if collection.count() > 0:
        return
    docs = [item["text"] for item in KB_ITEMS]
    ids = [item["id"] for item in KB_ITEMS]
    metadatas = [{"title": item["title"]} for item in KB_ITEMS]

    embeddings = embed_model.encode(docs).tolist()
    collection.add(documents=docs, ids=ids, metadatas=metadatas, embeddings=embeddings)
    logger.info(f"Seeded {len(docs)} documents indexed with domain embeddings.")

seed_database()

class DomainQueryRequest(BaseModel):
    question: str = Field(..., example="What consensus protocol does HyperionDB use?")

class DomainQueryResponse(BaseModel):
    question: str
    matched_doc_id: str
    title: str
    answer: str

@app.post("/api/v1/domain-query", response_model=DomainQueryResponse)
async def domain_rag_query(payload: DomainQueryRequest):
    q_str = payload.question.strip()
    if not q_str:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    q_vec = embed_model.encode(q_str).tolist()
    res = collection.query(query_embeddings=[q_vec], n_results=1)

    doc_id = res["ids"][0][0]
    matched_text = res["documents"][0][0]
    matched_title = res["metadatas"][0][0]["title"]

    prompt = (
        f"Context:\n{matched_text}\n\n"
        f"Question: {q_str}\n"
        f"Answer directly in one concise sentence:"
    )

    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=100
    )
    answer = resp.choices[0].message.content.strip()

    return DomainQueryResponse(
        question=q_str,
        matched_doc_id=doc_id,
        title=matched_title,
        answer=answer
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
