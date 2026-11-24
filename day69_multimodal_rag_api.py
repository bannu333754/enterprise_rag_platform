import time
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [MULTIMODAL] %(message)s")
logger = logging.getLogger("MultiModalRAG")

app = FastAPI(
    title="Day 69 Enterprise Multi-Modal RAG Service",
    version="12.0.0",
    description="Multi-modal document processing for text, structured tables, and visual diagram descriptions"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day69_multimodal_vault")

groq_client = Groq(api_key="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")

# ----------------- Seed Multi-Modal Knowledge Corpus -----------------
# Demonstrating text, tabular financial data, and visual architecture blueprints
CORPUS_ITEMS = [
    {
        "id": "text_q3_exec_summary",
        "type": "text",
        "search_content": "Executive Summary: Cloud infrastructure segment grew significantly in fiscal year 2026 driven by enterprise AI adoption and managed Kubernetes services.",
        "payload": "Executive Summary: Cloud infrastructure segment grew significantly in fiscal year 2026 driven by enterprise AI adoption and managed Kubernetes services."
    },
    {
        "id": "table_q3_financial_breakdown",
        "type": "table",
        "search_content": "Financial table quarterly revenue operating income earnings per share EPS for Q1 Q2 and Q3 2026. Cloud revenue and hardware sales metrics.",
        "payload": (
            "| Quarter | Revenue ($M) | Operating Income ($M) | EPS ($) |\n"
            "|:-------:|:------------:|:--------------------:|:-------:|\n"
            "| Q1 2026 | 4,200        | 1,150                | 2.15    |\n"
            "| Q2 2026 | 4,850        | 1,320                | 2.48    |\n"
            "| Q3 2026 | 5,600        | 1,590                | 3.12    |"
        )
    },
    {
        "id": "image_diagram_zero_trust",
        "type": "image_diagram",
        "search_content": "Architecture schematic diagram showing Zero Trust Network Access ZTNA with Envoy Gateway mTLS encryption and HashiCorp Vault token exchange.",
        "payload": (
            "[Visual Blueprint: Zero-Trust Security Ingress]\n"
            "- Component A: Client enters via Cloudflare edge CDN\n"
            "- Component B: Envoy Gateway inspects SPIFFE identity certificate\n"
            "- Component C: HashiCorp Vault verifies dynamic mTLS certs before forwarding to pod service mesh\n"
            "- Protocol: Strict wire encryption with ChaCha20-Poly1305 on port 8443."
        )
    }
]

def seed_multimodal_db():
    if collection.count() > 0:
        return

    documents = [item["search_content"] for item in CORPUS_ITEMS]
    ids = [item["id"] for item in CORPUS_ITEMS]
    metadatas = [{"type": item["type"], "payload": item["payload"]} for item in CORPUS_ITEMS]

    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    logger.info(f"Seeded {len(documents)} multi-modal objects into ChromaDB.")

seed_multimodal_db()

# ----------------- Request / Response Models -----------------
class MultiModalQueryRequest(BaseModel):
    question: str = Field(..., example="What was the EPS reported in Q3 2026?")
    filter_type: Optional[str] = Field(default=None, example="table")  # text | table | image_diagram | None

class RetrievedContextItem(BaseModel):
    id: str
    type: str
    payload: str

class MultiModalQueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_items: List[RetrievedContextItem]
    detected_types: List[str]

# ----------------- API Endpoints -----------------
@app.post("/api/v1/multimodal-query", response_model=MultiModalQueryResponse)
async def multimodal_rag_query(payload: MultiModalQueryRequest):
    q_str = payload.question.strip()
    if not q_str:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    q_vec = embed_model.encode(q_str).tolist()

    # Optional metadata filtering by modality
    where_filter = {"type": payload.filter_type} if payload.filter_type else None

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=2,
        where=where_filter
    )

    retrieved_metas = results["metadatas"][0] if results["metadatas"] else []
    retrieved_ids = results["ids"][0] if results["ids"] else []

    context_snippets = []
    items: List[RetrievedContextItem] = []
    detected_types = []

    for item_id, meta in zip(retrieved_ids, retrieved_metas):
        m_type = meta["type"]
        m_payload = meta["payload"]
        detected_types.append(m_type)
        items.append(RetrievedContextItem(id=item_id, type=m_type, payload=m_payload))
        context_snippets.append(f"[{m_type.upper()} CONTENT from {item_id}]:\n{m_payload}")

    combined_context = "\n\n".join(context_snippets)

    # Multi-Modal LLM Synthesis
    prompt = (
        f"You are an enterprise AI reasoning over multi-modal inputs including tables, diagrams, and text.\n"
        f"Context:\n{combined_context}\n\n"
        f"Question: {q_str}\n"
        f"Answer the question directly and concisely in 1-2 factual sentences:"
    )

    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=150
    )
    answer = resp.choices[0].message.content

    return MultiModalQueryResponse(
        question=q_str,
        answer=answer,
        retrieved_items=items,
        detected_types=list(set(detected_types))
    )
