import time
import math
import logging
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi
from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [HYBRID] %(message)s")
logger = logging.getLogger("HybridSearchEngine")

app = FastAPI(
    title="Day 66 Enterprise Dense-Sparse Hybrid Search Engine",
    version="9.0.0",
    description="Dense embeddings + BM25 Lexical Keyword search combined with Reciprocal Rank Fusion (RRF)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Corpus Documents -----------------
CORPUS_DOCUMENTS = [
    {
        "id": "doc_1",
        "text": "The distributed database encountered system failure with code ERR_K8S_503 during pod rolling upgrade.",
        "category": "incident_logs"
    },
    {
        "id": "doc_2",
        "text": "High availability cluster configuration requires at least three master nodes and etcd raft consensus.",
        "category": "infrastructure"
    },
    {
        "id": "doc_3",
        "text": "When microservices fail with HTTP 503 gateway timeout, verify ingress controller upstream timeouts and proxy limits.",
        "category": "troubleshooting"
    },
    {
        "id": "doc_4",
        "text": "OpenTelemetry collector exports telemetry metrics and distributed spans to Prometheus and Jaeger endpoints.",
        "category": "observability"
    },
    {
        "id": "doc_5",
        "text": "For bug ERR_K8S_503, apply patch version v1.28.4 to resolve memory leak in the core scheduling daemon.",
        "category": "patch_notes"
    }
]

# ----------------- Dense Vector Store Setup -----------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day66_hybrid_rag")

if collection.count() == 0:
    collection.add(
        documents=[doc["text"] for doc in CORPUS_DOCUMENTS],
        ids=[doc["id"] for doc in CORPUS_DOCUMENTS],
        metadatas=[{"category": doc["category"]} for doc in CORPUS_DOCUMENTS]
    )
    logger.info("ChromaDB dense index initialized.")

# ----------------- Sparse BM25 Index Setup -----------------
def tokenize(text: str) -> List[str]:
    # Lexical tokenization: lowercase, split punctuation & alphanumeric
    cleaned = "".join([c.lower() if c.isalnum() or c.isspace() or c == "_" else " " for c in text])
    return [token for token in cleaned.split() if token]

tokenized_corpus = [tokenize(doc["text"]) for doc in CORPUS_DOCUMENTS]
bm25_index = BM25Okapi(tokenized_corpus)
logger.info("BM25 sparse index initialized.")

# ----------------- Reciprocal Rank Fusion (RRF) -----------------
def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]], 
    sparse_results: List[Dict[str, Any]], 
    k: int = 60,
    top_n: int = 3
) -> List[Dict[str, Any]]:
    """
    Combines dense and sparse ranked lists using Reciprocal Rank Fusion (RRF).
    RRF Score(d) = sum( 1 / (k + rank) )
    """
    rrf_scores: Dict[str, float] = {}
    doc_details: Dict[str, Dict[str, Any]] = {}
    rank_metrics: Dict[str, Dict[str, int]] = {}

    # 1. Process Dense Rankings
    for rank, item in enumerate(dense_results, start=1):
        doc_id = item["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        doc_details[doc_id] = item
        if doc_id not in rank_metrics:
            rank_metrics[doc_id] = {}
        rank_metrics[doc_id]["dense_rank"] = rank

    # 2. Process Sparse Rankings
    for rank, item in enumerate(sparse_results, start=1):
        doc_id = item["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        doc_details[doc_id] = item
        if doc_id not in rank_metrics:
            rank_metrics[doc_id] = {}
        rank_metrics[doc_id]["sparse_rank"] = rank

    # 3. Sort by aggregated RRF score descending
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    fused_results = []
    for doc_id, score in sorted_docs[:top_n]:
        res = {
            "id": doc_id,
            "text": doc_details[doc_id]["text"],
            "rrf_score": round(score, 6),
            "dense_rank": rank_metrics[doc_id].get("dense_rank", "N/A"),
            "sparse_rank": rank_metrics[doc_id].get("sparse_rank", "N/A"),
        }
        fused_results.append(res)

    return fused_results

groq_client = Groq(api_key="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")

# ----------------- Request / Response Models -----------------
class HybridQueryRequest(BaseModel):
    question: str = Field(..., example="How do I resolve ERR_K8S_503 error?")
    top_k: int = Field(default=3, ge=1, le=5)

class HybridQueryResponse(BaseModel):
    question: str
    answer: str
    retrieval_mode: str
    fused_documents: List[Dict[str, Any]]

# ----------------- API Endpoints -----------------
@app.post("/api/v1/hybrid-search", response_model=HybridQueryResponse)
async def hybrid_search_query(payload: HybridQueryRequest):
    query_text = payload.question.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Dense Vector Search
    q_vec = embed_model.encode(query_text).tolist()
    dense_raw = collection.query(query_embeddings=[q_vec], n_results=len(CORPUS_DOCUMENTS))
    dense_results = [
        {"id": doc_id, "text": text}
        for doc_id, text in zip(dense_raw["ids"][0], dense_raw["documents"][0])
    ]

    # 2. Sparse BM25 Search
    tokenized_query = tokenize(query_text)
    bm25_scores = bm25_index.get_scores(tokenized_query)
    
    # Pair scores with document metadata and sort descending
    indexed_scores = list(enumerate(bm25_scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    sparse_results = [
        {"id": CORPUS_DOCUMENTS[idx]["id"], "text": CORPUS_DOCUMENTS[idx]["text"]}
        for idx, score in indexed_scores if score > 0
    ]

    # 3. Reciprocal Rank Fusion
    fused_docs = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=payload.top_k)

    # 4. LLM Synthesis with Fused Context
    context_str = "\n".join([f"- {d['text']}" for d in fused_docs])
    prompt = (
        f"Context from internal documentation:\n{context_str}\n\n"
        f"Question: {query_text}\n"
        f"Provide an accurate and actionable answer based strictly on the context:"
    )

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200
    )
    answer = response.choices[0].message.content

    return HybridQueryResponse(
        question=query_text,
        answer=answer,
        retrieval_mode="Dense_Sparse_Hybrid_RRF",
        fused_documents=fused_docs
    )
