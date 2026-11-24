import time
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

# OpenTelemetry Tracing Core
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace import Status, StatusCode

# ----------------- OpenTelemetry Tracer Setup -----------------
resource = Resource.create({"service.name": "rag-enterprise-query-service", "deployment.environment": "production"})
provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("rag.tracing.tracer", "7.0.0")
propagator = TraceContextTextMapPropagator()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [TRACE] %(message)s")
logger = logging.getLogger("DistributedTracing")

app = FastAPI(
    title="Day 63 Enterprise Distributed Tracing Service",
    version="7.0.0",
    description="End-to-end W3C traceparent propagation and OpenTelemetry span hierarchies"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Vector Store & Ingestion -----------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day63_tracing_rag")

if collection.count() == 0:
    collection.add(
        documents=[
            "Distributed tracing tracks request flows across decoupled microservice networks via unique trace identifiers.",
            "OpenTelemetry provides a vendor-agnostic specification and SDKs for generating metrics, logs, and trace spans.",
            "Jaeger is an open-source distributed tracing system inspired by Google Dapper used to monitor microservices."
        ],
        ids=["doc_1", "doc_2", "doc_3"],
        metadatas=[{"source": "tracing_standard"}, {"source": "otel_docs"}, {"source": "jaeger_docs"}]
    )

groq_client = Groq(api_key="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")

class QueryPayload(BaseModel):
    question: str = Field(..., example="How does distributed tracing track microservice requests?")

# ----------------- Traced RAG Pipeline Endpoint -----------------
@app.post("/api/v1/traced-query")
async def execute_traced_query(request: Request, payload: QueryPayload):
    # Extract incoming W3C traceparent context from HTTP headers if present
    extracted_context = propagator.extract(carrier=dict(request.headers))

    with tracer.start_as_current_span("rag.pipeline.execute", context=extracted_context) as root_span:
        trace_id = format(root_span.get_span_context().trace_id, "032x")
        span_id = format(root_span.get_span_context().span_id, "016x")

        root_span.set_attribute("rag.question", payload.question)
        root_span.set_attribute("http.client_ip", request.client.host if request.client else "unknown")

        logger.info(f"Incoming Request | Trace ID: {trace_id} | Root Span ID: {span_id}")

        # Sub-Span 1: Query Embedding
        with tracer.start_as_current_span("embed.generate") as embed_span:
            start_embed = time.perf_counter()
            q_vec = embed_model.encode(payload.question).tolist()
            embed_duration = (time.perf_counter() - start_embed) * 1000

            embed_span.set_attribute("embedding.model", "all-MiniLM-L6-v2")
            embed_span.set_attribute("embedding.dimensions", len(q_vec))
            embed_span.set_attribute("embedding.duration_ms", embed_duration)

        # Sub-Span 2: Vector Search
        with tracer.start_as_current_span("vector.retrieval") as db_span:
            start_db = time.perf_counter()
            results = collection.query(query_embeddings=[q_vec], n_results=2)
            db_duration = (time.perf_counter() - start_db) * 1000

            retrieved_docs = results["documents"][0] if results["documents"] else []
            retrieved_ids = results["ids"][0] if results["ids"] else []

            db_span.set_attribute("db.system", "chromadb")
            db_span.set_attribute("db.retrieved_count", len(retrieved_docs))
            db_span.set_attribute("db.retrieved_ids", ",".join(retrieved_ids))
            db_span.set_attribute("db.duration_ms", db_duration)

        # Sub-Span 3: LLM Synthesis
        with tracer.start_as_current_span("llm.synthesis") as llm_span:
            start_llm = time.perf_counter()
            context = " ".join(retrieved_docs)
            prompt = f"Context: {context}\n\nQuestion: {payload.question}\nProvide a concise 1-sentence answer:"

            try:
                resp = groq_client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=150
                )
                answer = resp.choices[0].message.content
                llm_duration = (time.perf_counter() - start_llm) * 1000

                llm_span.set_attribute("llm.model", "openai/gpt-oss-20b")
                llm_span.set_attribute("llm.token_count", len(answer.split()))
                llm_span.set_attribute("llm.duration_ms", llm_duration)
                llm_span.set_status(Status(StatusCode.OK))
            except Exception as e:
                llm_span.record_exception(e)
                llm_span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

        root_span.set_status(Status(StatusCode.OK))

        response_data = {
            "answer": answer,
            "telemetry": {
                "trace_id": trace_id,
                "root_span_id": span_id,
                "embed_ms": round(embed_duration, 2),
                "db_ms": round(db_duration, 2),
                "llm_ms": round(llm_duration, 2),
                "total_ms": round(embed_duration + db_duration + llm_duration, 2)
            }
        }

        # Inject outgoing traceparent header for client propagation
        carrier = {}
        propagator.inject(carrier=carrier)
        headers = {"traceparent": carrier.get("traceparent", f"00-{trace_id}-{span_id}-01")}

        return Response(
            content=str(response_data).replace("'", "\""),
            media_type="application/json",
            headers=headers
        )
