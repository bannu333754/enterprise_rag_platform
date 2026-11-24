import time
import hashlib
import logging
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
import chromadb
from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [CACHE] %(message)s")
logger = logging.getLogger("SemanticCache")

app = FastAPI(
    title="Day 61 Enterprise Semantic Caching Core",
    version="6.0.0",
    description="Sub-10ms query resolution using embedding similarity caching, LRU eviction, and exact match keys"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Semantic Cache Store Implementation -----------------
class CacheEntry:
    def __init__(self, query: str, embedding: List[float], response: str, ttl_seconds: float = 300.0):
        self.query = query
        self.embedding = embedding
        self.response = response
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.ttl_seconds = ttl_seconds
        self.hit_count = 0

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds

    def touch(self):
        self.last_accessed = time.time()
        self.hit_count += 1

class SemanticCache:
    def __init__(self, capacity: int = 50, similarity_threshold: float = 0.80, ttl_seconds: float = 300.0):
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.exact_cache: Dict[str, CacheEntry] = {}
        self.entries: List[CacheEntry] = []

    def _hash_query(self, query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()

    def _evict_if_needed(self):
        # 1. Purge expired entries
        now = time.time()
        valid_entries = []
        for e in self.entries:
            if not e.is_expired():
                valid_entries.append(e)
            else:
                key = self._hash_query(e.query)
                self.exact_cache.pop(key, None)
        self.entries = valid_entries

        # 2. LRU eviction if over capacity
        if len(self.entries) >= self.capacity:
            # Sort by least recently accessed
            self.entries.sort(key=lambda x: x.last_accessed)
            evicted = self.entries.pop(0)
            self.exact_cache.pop(self._hash_query(evicted.query), None)
            logger.info(f"Evicted LRU cache entry: '{evicted.query}'")

    def lookup(self, query: str, query_embedding) -> Tuple[Optional[str], str, float]:
        """
        Returns (response, match_type, score)
        match_type: "EXACT_HIT", "SEMANTIC_HIT", or "MISS"
        """
        self._evict_if_needed()
        key = self._hash_query(query)

        # Level 1: Exact Hash Match
        if key in self.exact_cache:
            entry = self.exact_cache[key]
            if not entry.is_expired():
                entry.touch()
                return entry.response, "EXACT_HIT", 1.0

        if not self.entries:
            return None, "MISS", 0.0

        # Level 2: Semantic Vector Cosine Match
        all_embeddings = [e.embedding for e in self.entries]
        scores = util.cos_sim(query_embedding, all_embeddings)[0].tolist()

        best_score = -1.0
        best_idx = -1
        for idx, score in enumerate(scores):
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_score >= self.similarity_threshold:
            matched_entry = self.entries[best_idx]
            matched_entry.touch()
            # Link exact key for faster retrieval on next identical lookup
            self.exact_cache[key] = matched_entry
            return matched_entry.response, "SEMANTIC_HIT", best_score

        return None, "MISS", best_score if best_score != -1.0 else 0.0

    def put(self, query: str, query_embedding: List[float], response: str):
        self._evict_if_needed()
        entry = CacheEntry(query, query_embedding, response, self.ttl_seconds)
        self.entries.append(entry)
        self.exact_cache[self._hash_query(query)] = entry
        logger.info(f"Saved to semantic cache: '{query}' (Total entries: {len(self.entries)})")

semantic_cache = SemanticCache(capacity=50, similarity_threshold=0.80, ttl_seconds=300.0)

# ----------------- Model & Store Initialization -----------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day61_caching_rag")

if collection.count() == 0:
    collection.add(
        documents=[
            "Redis is an open-source, in-memory data structure store used as a database, cache, and message broker.",
            "Semantic caching reduces LLM cost and latency by returning cached responses for semantically equivalent queries.",
            "Cosine similarity measures the angle between two multi-dimensional dense vectors to quantify semantic relatedness."
        ],
        ids=["doc_1", "doc_2", "doc_3"],
        metadatas=[{"source": "redis_docs"}, {"source": "cache_paper"}, {"source": "math_foundations"}]
    )

groq_client = Groq(api_key="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")

# ----------------- Request / Response Models -----------------
class QueryRequest(BaseModel):
    question: str = Field(..., example="What is Redis used for?")

class QueryResponse(BaseModel):
    answer: str
    cache_status: str
    similarity_score: float
    latency_ms: float
    tokens_saved: int

# ----------------- API Endpoints -----------------
@app.post("/api/v1/cached-query", response_model=QueryResponse)
async def cached_query(payload: QueryRequest):
    start_time = time.perf_counter()
    q_str = payload.question.strip()

    if not q_str:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Compute query vector
    q_vec = embed_model.encode(q_str)

    # 2. Check Semantic In-Memory Cache
    cached_resp, match_type, score = semantic_cache.lookup(q_str, q_vec)

    if cached_resp:
        total_latency = (time.perf_counter() - start_time) * 1000
        logger.info(f"[{match_type}] '{q_str}' -> Latency: {total_latency:.2f}ms (Score: {score:.4f})")
        return QueryResponse(
            answer=cached_resp,
            cache_status=match_type,
            similarity_score=round(score, 4),
            latency_ms=round(total_latency, 2),
            tokens_saved=len(cached_resp.split())
        )

    # 3. Cache Miss: Execute standard vector search + LLM synthesis
    q_vec_list = q_vec.tolist()
    results = collection.query(query_embeddings=[q_vec_list], n_results=2)
    retrieved_docs = results["documents"][0] if results["documents"] else []
    context = " ".join(retrieved_docs)

    prompt = (
        f"Context: {context}\n\n"
        f"Question: {q_str}\n"
        f"Provide a concise, factual answer:"
    )

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=150
    )
    answer = response.choices[0].message.content

    # 4. Save to Semantic Cache
    semantic_cache.put(q_str, q_vec_list, answer)

    total_latency = (time.perf_counter() - start_time) * 1000
    return QueryResponse(
        answer=answer,
        cache_status="CACHE_MISS",
        similarity_score=round(score, 4),
        latency_ms=round(total_latency, 2),
        tokens_saved=0
    )

@app.get("/api/v1/cache/stats")
def cache_stats():
    return {
        "total_entries": len(semantic_cache.entries),
        "exact_keys": len(semantic_cache.exact_cache),
        "capacity": semantic_cache.capacity,
        "threshold": semantic_cache.similarity_threshold,
        "active_items": [
            {"query": e.query, "hits": e.hit_count, "age_sec": round(time.time() - e.created_at, 1)}
            for e in semantic_cache.entries
        ]
    }
