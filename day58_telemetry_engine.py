import time
import json
import asyncio
from typing import Dict, List
from fastapi import FastAPI, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from groq import Groq

app = FastAPI(
    title="Day 58 Enterprise Observability Core",
    version="3.0.0",
    description="FastAPI service instrumented with Prometheus metrics and custom OpenTelemetry trace spans"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Prometheus Metrics Definitions -----------------
REQUEST_COUNT = Counter(
    "rag_http_requests_total",
    "Total incoming HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "Overall HTTP request duration in seconds",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

RETRIEVAL_LATENCY = Histogram(
    "rag_vector_retrieval_seconds",
    "Latency of ChromaDB query vector retrieval",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)

RERANK_LATENCY = Histogram(
    "rag_cross_encoder_rerank_seconds",
    "Latency of Cross-Encoder re-ranking step",
    buckets=[0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
)

TOKENS_GENERATED = Counter(
    "rag_llm_tokens_generated_total",
    "Total tokens streamed by LLM",
    ["model"]
)

ACTIVE_SESSIONS = Gauge(
    "rag_active_streaming_sessions",
    "Current active streaming user sessions"
)

# ----------------- Core Models & Clients -----------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day58_telemetry_rag")

# Seed mock baseline documents if empty
if collection.count() == 0:
    collection.add(
        documents=[
            "Enterprise Observability requires metric aggregation, trace context propagation, and structured log capture.",
            "Prometheus collects metrics from monitored targets by scraping metrics HTTP endpoints at designated intervals.",
            "Grafana integrates with Prometheus to render time-series visualization panels and alert management rules."
        ],
        ids=["doc_1", "doc_2", "doc_3"],
        metadatas=[{"source": "observability_whitepaper.pdf", "page": 1},
                   {"source": "prometheus_guide.pdf", "page": 4},
                   {"source": "grafana_docs.pdf", "page": 2}]
    )

groq_client = Groq(api_key="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")

# ----------------- Prometheus Metrics Scrape Endpoint -----------------
@app.get("/metrics")
def metrics():
    """Prometheus metrics scraper endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return {"status": "healthy", "service": "day58-observability-engine"}

@app.post("/api/v1/query")
async def query_monitored_rag(request: Request):
    """
    Instrumented RAG query endpoint with granular tracing & metric recording.
    """
    start_total = time.perf_counter()
    status_label = "200"

    try:
        body = await request.json()
        question = body.get("question", "").strip()

        if not question:
            status_label = "400"
            return Response(content=json.dumps({"error": "Empty question"}), status_code=400)

        # Trace Span 1: Vector Space Retrieval
        t_ret_start = time.perf_counter()
        q_vec = embed_model.encode(question).tolist()
        results = collection.query(query_embeddings=[q_vec], n_results=3)
        candidates = results["documents"][0] if results["documents"] else []
        retrieval_duration = time.perf_counter() - t_ret_start
        RETRIEVAL_LATENCY.observe(retrieval_duration)

        # Trace Span 2: Cross-Encoder Re-Ranking
        t_rerank_start = time.perf_counter()
        top_chunks = []
        if candidates:
            pairs = [[question, doc] for doc in candidates]
            scores = reranker.predict(pairs)
            best_idx = int(scores.argmax())
            top_chunks = [candidates[best_idx]]
        rerank_duration = time.perf_counter() - t_rerank_start
        RERANK_LATENCY.observe(rerank_duration)

        # Trace Span 3: LLM Generation
        ACTIVE_SESSIONS.inc()
        context_str = "\n".join(top_chunks) if top_chunks else "No context available."
        prompt = f"Context: {context_str}\n\nQuestion: {question}\nAnswer clearly:"

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250
        )
        answer = response.choices[0].message.content
        token_count = len(answer.split())
        TOKENS_GENERATED.labels(model="gpt-oss-20b").inc(token_count)
        ACTIVE_SESSIONS.dec()

        total_latency = time.perf_counter() - start_total
        REQUEST_LATENCY.labels(endpoint="/api/v1/query").observe(total_latency)
        REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/query", status=status_label).inc()

        return {
            "answer": answer,
            "telemetry": {
                "retrieval_ms": round(retrieval_duration * 1000, 2),
                "rerank_ms": round(rerank_duration * 1000, 2),
                "total_pipeline_ms": round(total_latency * 1000, 2),
                "tokens_streamed": token_count
            }
        }

    except Exception as e:
        status_label = "500"
        REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/query", status=status_label).inc()
        return Response(content=json.dumps({"error": str(e)}), status_code=500)
