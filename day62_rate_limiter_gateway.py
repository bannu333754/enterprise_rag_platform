import time
import math
import logging
from typing import Dict, Tuple, Optional
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [GATEWAY] %(message)s")
logger = logging.getLogger("RateLimiterGateway")

app = FastAPI(
    title="Day 62 Enterprise RAG API Gateway & Rate Limiter",
    version="7.0.0",
    description="Token-bucket rate limiting, burst management, and token expenditure quotas"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Client Tiers & Quota Configurations -----------------
TIER_POLICIES = {
    "free": {
        "capacity": 5,             # Max burst tokens
        "refill_rate_per_sec": 0.2, # 1 token every 5 seconds (12 req/min)
        "max_token_budget_hourly": 2000
    },
    "enterprise": {
        "capacity": 30,            # Max burst tokens
        "refill_rate_per_sec": 2.0, # 2 tokens per second (120 req/min)
        "max_token_budget_hourly": 100000
    }
}

API_KEYS_DB = {
    "key_free_user": {"tier": "free", "client_name": "Acme Free Tier Client"},
    "key_enterprise_corp": {"tier": "enterprise", "client_name": "Globex Enterprise Partner"}
}

# ----------------- Token Bucket Implementation -----------------
class TokenBucket:
    def __init__(self, capacity: float, refill_rate_per_sec: float):
        self.capacity = capacity
        self.refill_rate = refill_rate_per_sec
        self.tokens = capacity
        self.last_update = time.time()
        self.llm_tokens_consumed = 0

    def consume(self, amount: float = 1.0) -> Tuple[bool, int, float]:
        """
        Returns (is_allowed, remaining_tokens, retry_after_seconds)
        """
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now

        # Replenish bucket based on elapsed time
        self.tokens = min(self.capacity, self.tokens + (elapsed * self.refill_rate))

        if self.tokens >= amount:
            self.tokens -= amount
            return True, int(self.tokens), 0.0

        # Calculate wait time needed for at least 'amount' tokens
        deficit = amount - self.tokens
        retry_after = deficit / self.refill_rate
        return False, int(self.tokens), math.ceil(retry_after)

CLIENT_BUCKETS: Dict[str, TokenBucket] = {}

def get_or_create_bucket(client_id: str, tier: str) -> TokenBucket:
    if client_id not in CLIENT_BUCKETS:
        policy = TIER_POLICIES.get(tier, TIER_POLICIES["free"])
        CLIENT_BUCKETS[client_id] = TokenBucket(
            capacity=policy["capacity"],
            refill_rate_per_sec=policy["refill_rate_per_sec"]
        )
    return CLIENT_BUCKETS[client_id]

# ----------------- Models & Knowledge Base -----------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day62_gateway_rag")

if collection.count() == 0:
    collection.add(
        documents=[
            "Rate limiting protects backend systems from Denial of Service (DoS) attacks and ensures fair API distribution.",
            "The Token Bucket algorithm allows brief bursts of traffic while enforcing a strict average request rate.",
            "HTTP 429 Too Many Requests indicates the user has sent too many requests in a given amount of time."
        ],
        ids=["doc_1", "doc_2", "doc_3"],
        metadatas=[{"topic": "security"}, {"topic": "algorithms"}, {"topic": "http_standards"}]
    )

groq_client = Groq(api_key="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")

class QueryPayload(BaseModel):
    question: str = Field(..., example="What is the Token Bucket algorithm?")

# ----------------- Rate-Limited Query Endpoint -----------------
@app.post("/api/v1/throttled-query")
async def throttled_rag_query(request: Request, payload: QueryPayload):
    # 1. Resolve Client Identity via Header or Fallback to IP
    api_key = request.headers.get("X-API-Key")
    client_ip = request.client.host if request.client else "127.0.0.1"

    if api_key and api_key in API_KEYS_DB:
        client_id = api_key
        tier = API_KEYS_DB[api_key]["tier"]
    else:
        client_id = f"ip_{client_ip}"
        tier = "free"

    bucket = get_or_create_bucket(client_id, tier)
    allowed, remaining, retry_after = bucket.consume(1.0)

    # Standard rate limit response headers
    headers = {
        "X-RateLimit-Limit": str(int(bucket.capacity)),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Tier": tier
    }

    # 2. Enforce 429 Rejection if Token Bucket is Dry
    if not allowed:
        headers["Retry-After"] = str(int(retry_after))
        logger.warning(f"Rate limit exceeded for client '{client_id}'. Retry-After: {retry_after}s")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for {tier} tier. Please retry after {int(retry_after)} seconds.",
            headers=headers
        )

    # 3. Vector Retrieval & LLM Generation
    question = payload.question.strip()
    q_vec = embed_model.encode(question).tolist()
    results = collection.query(query_embeddings=[q_vec], n_results=2)
    context = " ".join(results["documents"][0]) if results["documents"] else ""

    prompt = f"Context: {context}\n\nQuestion: {question}\nConcise answer in one sentence:"
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=100
    )
    answer = response.choices[0].message.content
    tokens_used = len(answer.split())
    bucket.llm_tokens_consumed += tokens_used

    return Response(
        content=f'{{"answer": "{answer}", "tokens_used": {tokens_used}, "tier": "{tier}", "remaining_burst_tokens": {remaining}}}',
        media_type="application/json",
        headers=headers
    )

@app.get("/api/v1/gateway/status")
def gateway_status():
    return {
        "active_tracked_clients": len(CLIENT_BUCKETS),
        "clients": {
            cid: {
                "current_tokens": round(b.tokens, 2),
                "capacity": b.capacity,
                "total_llm_tokens": b.llm_tokens_consumed
            }
            for cid, b in CLIENT_BUCKETS.items()
        }
    }
