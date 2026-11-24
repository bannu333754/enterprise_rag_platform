import time
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, Depends, HTTPException, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

# Configure structured audit logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [AUDIT] %(message)s")
logger = logging.getLogger("RAG_Security_Audit")

app = FastAPI(
    title="Day 59 Multi-Tenant RBAC Secure RAG Engine",
    version="4.0.0",
    description="Vector search secured with tenant isolation, clearance tiers, and dynamic metadata filtering"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Security Model & Clearance Tiers -----------------
CLEARANCE_HIERARCHY = {
    "public": 1,
    "internal": 2,
    "confidential": 3,
    "restricted": 4
}

class UserContext(BaseModel):
    user_id: str
    tenant_id: str
    roles: List[str]
    clearance_level: str

class QueryRequest(BaseModel):
    question: str = Field(..., example="What are our financial targets?")
    top_k: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    answer: str
    tenant_id: str
    effective_clearance: str
    retrieved_count: int
    sources: List[Dict]

# ----------------- Mock Identity Provider (JWT Simulation) -----------------
MOCK_USER_TOKENS: Dict[str, UserContext] = {
    "token_alice_acme": UserContext(
        user_id="alice_dev",
        tenant_id="acme_corp",
        roles=["engineer"],
        clearance_level="internal"
    ),
    "token_bob_acme": UserContext(
        user_id="bob_exec",
        tenant_id="acme_corp",
        roles=["executive"],
        clearance_level="restricted"
    ),
    "token_charlie_globex": UserContext(
        user_id="charlie_pm",
        tenant_id="globex_inc",
        roles=["product_manager"],
        clearance_level="confidential"
    )
}

def authenticate_user(authorization: Optional[str] = Header(None)) -> UserContext:
    """Extract and validate bearer token claims."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Bearer token authorization header"
        )
    token = authorization.split(" ")[1]
    user_ctx = MOCK_USER_TOKENS.get(token)
    if not user_ctx:
        logger.warning(f"Failed authentication attempt with invalid token: {token}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired authentication credentials"
        )
    return user_ctx

# ----------------- Vector Store & Ingestion -----------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day59_multitenant_rag")

def seed_multitenant_data():
    """Populate partitioned documents across multiple tenants and clearances."""
    if collection.count() > 0:
        return

    documents = [
        # Acme Corp documents
        "Acme Corp public API documentation: developers can integrate with REST endpoints.",
        "Acme Corp internal handbook: sprint planning happens every second Monday at 10 AM.",
        "Acme Corp confidential roadmaps: Q3 AI agent deployment target is October 2026.",
        "Acme Corp restricted M&A: Acme is evaluating acquisition of BetaCorp for 45M USD.",
        # Globex Inc documents
        "Globex Inc public mission: delivering scalable enterprise supply chain workflows.",
        "Globex Inc confidential finances: Q2 revenue reached 12.8M USD with 22 percent operating margin.",
        "Globex Inc restricted legal: ongoing settlement negotiations regarding proprietary patent."
    ]

    ids = [f"doc_{i}" for i in range(1, len(documents) + 1)]
    metadatas = [
        {"tenant_id": "acme_corp", "clearance": "public", "doc_type": "handbook"},
        {"tenant_id": "acme_corp", "clearance": "internal", "doc_type": "handbook"},
        {"tenant_id": "acme_corp", "clearance": "confidential", "doc_type": "strategy"},
        {"tenant_id": "acme_corp", "clearance": "restricted", "doc_type": "executive"},
        {"tenant_id": "globex_inc", "clearance": "public", "doc_type": "profile"},
        {"tenant_id": "globex_inc", "clearance": "confidential", "doc_type": "finance"},
        {"tenant_id": "globex_inc", "clearance": "restricted", "doc_type": "legal"}
    ]

    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    logger.info("Multi-tenant vector corpus seeded successfully.")

seed_multitenant_data()
groq_client = Groq(api_key="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")

# ----------------- Query Endpoint with Metadata Filtering -----------------
@app.post("/api/v1/secure-query", response_model=QueryResponse)
async def secure_rag_query(payload: QueryRequest, user: UserContext = Depends(authenticate_user)):
    user_clearance_rank = CLEARANCE_HIERARCHY.get(user.clearance_level, 1)
    
    # Calculate all clearance levels the user is authorized to view
    allowed_clearances = [
        level for level, rank in CLEARANCE_HIERARCHY.items()
        if rank <= user_clearance_rank
    ]

    # Synthesize dynamic ChromaDB security filter
    security_filter = {
        "$and": [
            {"tenant_id": {"$eq": user.tenant_id}},
            {"clearance": {"$in": allowed_clearances}}
        ]
    }

    logger.info(
        f"User: '{user.user_id}' | Tenant: '{user.tenant_id}' | "
        f"Clearance: '{user.clearance_level}' | Filter: {security_filter}"
    )

    # Vector similarity search with strict metadata filtering
    q_vec = embed_model.encode(payload.question).tolist()
    query_results = collection.query(
        query_embeddings=[q_vec],
        n_results=payload.top_k,
        where=security_filter
    )

    retrieved_docs = query_results["documents"][0] if query_results["documents"] else []
    retrieved_metas = query_results["metadatas"][0] if query_results["metadatas"] else []

    if not retrieved_docs:
        logger.warning(f"No documents matched for user '{user.user_id}' under current clearance.")
        return QueryResponse(
            answer="No accessible documents found matching your clearance and tenant boundaries.",
            tenant_id=user.tenant_id,
            effective_clearance=user.clearance_level,
            retrieved_count=0,
            sources=[]
        )

    # Synthesis with safe context
    context_str = "\n".join(retrieved_docs)
    prompt = (
        f"Context from internal documents:\n{context_str}\n\n"
        f"Question: {payload.question}\n"
        f"Provide a factual answer strictly based on the context:"
    )

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=250
    )
    answer = response.choices[0].message.content

    sources_info = [
        {"snippet": doc[:80] + "...", "clearance": meta.get("clearance"), "tenant": meta.get("tenant_id")}
        for doc, meta in zip(retrieved_docs, retrieved_metas)
    ]

    return QueryResponse(
        answer=answer,
        tenant_id=user.tenant_id,
        effective_clearance=user.clearance_level,
        retrieved_count=len(retrieved_docs),
        sources=sources_info
    )
