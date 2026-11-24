import time
import uuid
import logging
from typing import List, Dict, Any, Set
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [PARENT_RETRIEVER] %(message)s")
logger = logging.getLogger("ParentDocumentRetriever")

app = FastAPI(
    title="Day 68 Enterprise Multi-Vector Parent Document Retriever",
    version="11.0.0",
    description="Multi-vector indexing with child chunk vector search and parent document context expansion"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- In-Memory Parent DocStore -----------------
# In production, this can be backed by Redis or PostgreSQL
PARENT_DOCSTORE: Dict[str, Dict[str, Any]] = {}

# ----------------- Vector Store & Embedding Model -----------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
child_collection = chroma_client.get_or_create_collection(name="day68_child_vectors")

# ----------------- Sample Enterprise Parent Documents -----------------
KNOWLEDGE_BASE = [
    {
        "parent_id": "parent_vault_pki",
        "title": "HashiCorp Vault Enterprise PKI & Certificate Lifecycle Engine",
        "full_content": (
            "HashiCorp Vault acts as an automated Public Key Infrastructure (PKI) engine for enterprise clusters. "
            "It generates short-lived X.509 mutual TLS (mTLS) certificates on-the-fly, deprecating long-lived static secrets. "
            "The Vault intermediate certificate authority (CA) rotates dynamic certs with a default TTL of 72 hours. "
            "To prevent cert revocation list (CRL) bloat, Vault uses an internal delta-crl mechanism and OCSP responders. "
            "For disaster recovery, Vault utilizes automated Raft integrated storage snapshots replicated cross-region to S3 buckets. "
            "In case of a regional partition, unseal keys can be reconstructed using Shamir's secret sharing threshold (3 of 5 keys)."
        )
    },
    {
        "parent_id": "parent_kafka_tiered_storage",
        "title": "Apache Kafka 3.6 Tiered Storage Architecture & Offloading",
        "full_content": (
            "Apache Kafka Tiered Storage separates cluster computation from unbounded log retention storage. "
            "Active and recent hot segments remain on local low-latency NVMe drives for fast real-time consumer tail reads. "
            "Closed historical partition segments are asynchronously uploaded to Amazon S3 or Google Cloud Storage cold tier. "
            "The RemoteLogManager daemon handles the offloading cycle with a dedicated thread pool of 16 workers. "
            "Tiered storage reduces Kafka broker recovery time from multiple hours to less than 4 minutes after node reboot. "
            "Local disk storage utilization can be trimmed by up to 80% without altering standard Kafka consumer offset semantics."
        )
    }
]

def split_into_child_chunks(text: str, chunk_size: int = 25) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - 5
    return chunks

def seed_parent_child_pipeline():
    if child_collection.count() > 0:
        return

    child_docs = []
    child_ids = []
    child_metas = []

    for item in KNOWLEDGE_BASE:
        p_id = item["parent_id"]
        # Store large parent in Document Store
        PARENT_DOCSTORE[p_id] = {
            "title": item["title"],
            "full_content": item["full_content"]
        }

        # Split parent into small children for vector indexing
        children = split_into_child_chunks(item["full_content"])
        for idx, child_text in enumerate(children):
            c_id = f"{p_id}_c{idx}"
            child_docs.append(child_text)
            child_ids.append(c_id)
            child_metas.append({
                "parent_id": p_id,
                "child_index": idx,
                "child_text": child_text
            })

    child_collection.add(documents=child_docs, ids=child_ids, metadatas=child_metas)
    logger.info(f"Seeded {len(PARENT_DOCSTORE)} Parents and {len(child_docs)} Child vectors.")

seed_parent_child_pipeline()
groq_client = Groq(api_key="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")

# ----------------- Request / Response Models -----------------
class ParentQueryRequest(BaseModel):
    question: str = Field(..., example="What is the default TTL for Vault intermediate certificates?")
    top_k_children: int = Field(default=3, ge=1, le=10)

class RetrievedChildMeta(BaseModel):
    child_id: str
    parent_id: str
    child_snippet: str

class ParentQueryResponse(BaseModel):
    question: str
    answer: str
    matched_children_count: int
    resolved_parents_count: int
    matched_children: List[RetrievedChildMeta]
    resolved_parent_ids: List[str]

# ----------------- Endpoints -----------------
@app.post("/api/v1/parent-document-query", response_model=ParentQueryResponse)
async def query_parent_retriever(payload: ParentQueryRequest):
    q_str = payload.question.strip()
    if not q_str:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Search Vector DB for Child Chunks
    q_vec = embed_model.encode(q_str).tolist()
    search_res = child_collection.query(query_embeddings=[q_vec], n_results=payload.top_k_children)

    raw_metas = search_res["metadatas"][0] if search_res["metadatas"] else []
    raw_ids = search_res["ids"][0] if search_res["ids"] else []

    matched_children_list: List[RetrievedChildMeta] = []
    unique_parent_ids: Set[str] = set()

    for cid, meta in zip(raw_ids, raw_metas):
        pid = meta["parent_id"]
        unique_parent_ids.add(pid)
        matched_children_list.append(RetrievedChildMeta(
            child_id=cid,
            parent_id=pid,
            child_snippet=meta["child_text"][:80] + "..."
        ))

    # 2. Swap Child Chunks for Full Parent Documents from DocStore
    parent_contexts = []
    for pid in unique_parent_ids:
        if pid in PARENT_DOCSTORE:
            p_data = PARENT_DOCSTORE[pid]
            parent_contexts.append(f"Document [{p_data['title']}]:\n{p_data['full_content']}")

    full_context_str = "\n\n".join(parent_contexts)

    # 3. LLM Synthesis with Rich Parent Context
    prompt = (
        f"Context:\n{full_context_str}\n\n"
        f"Question: {q_str}\n"
        f"Provide a clear, factual answer in one concise sentence:"
    )

    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=150
    )
    answer = resp.choices[0].message.content

    return ParentQueryResponse(
        question=q_str,
        answer=answer,
        matched_children_count=len(matched_children_list),
        resolved_parents_count=len(unique_parent_ids),
        matched_children=matched_children_list,
        resolved_parent_ids=list(unique_parent_ids)
    )
