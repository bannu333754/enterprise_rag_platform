import time
import json
import logging
from enum import Enum
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
import chromadb
from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [SELF-HEAL] %(message)s")
logger = logging.getLogger("SelfHealingRAG")

app = FastAPI(
    title="Day 60 Autonomous Self-Healing RAG Engine",
    version="5.0.0",
    description="Enterprise RAG featuring Circuit Breakers, Dynamic DLQ, and Automated Self-Correction Loops"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Circuit Breaker Implementation -----------------
class CircuitState(str, Enum):
    CLOSED = "CLOSED"        # Normal operation
    OPEN = "OPEN"            # Fast-failing, downstream API protected
    HALF_OPEN = "HALF_OPEN"  # Testing downstream recovery

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_time_sec: float = 15.0):
        self.failure_threshold = failure_threshold
        self.recovery_time_sec = recovery_time_sec
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0

    def record_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"Circuit Breaker TRIPPED to OPEN! Failures: {self.failure_count}")

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_time_sec:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit Breaker transitioning from OPEN to HALF_OPEN")
                return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            return True
        return False

circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_time_sec=10.0)

# ----------------- Dead Letter Queue (DLQ) -----------------
DEAD_LETTER_QUEUE: List[Dict] = []

def send_to_dlq(query: str, reason: str, retrieved_context: List[str], attempts: int):
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "reason": reason,
        "retrieved_context": retrieved_context,
        "attempts": attempts
    }
    DEAD_LETTER_QUEUE.append(entry)
    logger.warning(f"Routed query to DLQ: '{query}' | Reason: {reason}")

# ----------------- Vector Store & Embedding Core -----------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day60_self_healing_rag")

if collection.count() == 0:
    collection.add(
        documents=[
            "Kubernetes Horizontal Pod Autoscalers dynamically scale replicas based on CPU and memory thresholds.",
            "Prometheus collects multi-dimensional time series metrics via HTTP pull scraping mechanisms.",
            "FastAPI leverages Python type hints and Pydantic validation to provide async-first REST performance."
        ],
        ids=["doc_1", "doc_2", "doc_3"],
        metadatas=[{"source": "k8s_docs"}, {"source": "prom_docs"}, {"source": "fastapi_docs"}]
    )

groq_client = Groq(api_key="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")

# ----------------- Self-Healing Critic & Schemas -----------------
class QueryInput(BaseModel):
    question: str = Field(..., example="How does HPA scale workloads?")
    simulate_failure: bool = Field(default=False, description="Forces downstream failure to test Circuit Breaker")

class QueryResult(BaseModel):
    answer: str
    status: str
    healed: bool
    remedial_attempts: int
    circuit_state: str
    context_used: List[str]

def evaluate_faithfulness(answer: str, context: List[str]) -> float:
    """Computes semantic alignment score between answer and retrieved context."""
    if not context:
        return 0.0
    ans_vec = embed_model.encode(answer)
    ctx_vec = embed_model.encode(" ".join(context))
    sim = float(util.cos_sim(ans_vec, ctx_vec)[0][0])
    return sim

# ----------------- Execution Endpoint -----------------
@app.post("/api/v1/resilient-query", response_model=QueryResult)
async def resilient_rag_query(payload: QueryInput):
    question = payload.question.strip()
    
    # 1. Circuit Breaker Check
    if not circuit_breaker.allow_request():
        send_to_dlq(question, "Circuit Breaker OPEN - Fast Fallback", [], 0)
        return QueryResult(
            answer="Service temporarily throttled due to upstream latency. Standard fallback: Refer to system documentation.",
            status="FALLBACK_CIRCUIT_OPEN",
            healed=False,
            remedial_attempts=0,
            circuit_state=circuit_breaker.state.value,
            context_used=[]
        )

    # 2. Simulated Downstream Fault Check (for testing)
    if payload.simulate_failure:
        circuit_breaker.record_failure()
        send_to_dlq(question, "Simulated Downstream Failure", [], 1)
        return QueryResult(
            answer="Downstream synthesis failed. Automated fallback answer served.",
            status="DOWNSTREAM_FAILED",
            healed=False,
            remedial_attempts=1,
            circuit_state=circuit_breaker.state.value,
            context_used=[]
        )

    attempts = 0
    max_retries = 2
    healed = False
    current_query = question

    while attempts <= max_retries:
        attempts += 1
        q_vec = embed_model.encode(current_query).tolist()
        results = collection.query(query_embeddings=[q_vec], n_results=2)
        retrieved_docs = results["documents"][0] if results["documents"] else []

        if not retrieved_docs:
            current_query = f"{question} infrastructure architecture"
            continue

        try:
            prompt = (
                f"Context: {' '.join(retrieved_docs)}\n\n"
                f"Question: {question}\n"
                f"Answer concisely in one factual sentence:"
            )
            resp = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150
            )
            synthesized_answer = resp.choices[0].message.content
            circuit_breaker.record_success()

            # Critic check: Groundedness & faithfulness verification
            faithfulness_score = evaluate_faithfulness(synthesized_answer, retrieved_docs)
            logger.info(f"Attempt {attempts} | Faithfulness Score: {faithfulness_score:.4f}")

            if faithfulness_score >= 0.45:
                return QueryResult(
                    answer=synthesized_answer,
                    status="SUCCESS",
                    healed=healed,
                    remedial_attempts=attempts - 1,
                    circuit_state=circuit_breaker.state.value,
                    context_used=retrieved_docs
                )
            else:
                logger.warning(f"Low faithfulness ({faithfulness_score:.4f}). Triggering self-healing rewrite...")
                healed = True
                current_query = f"Detailed overview of {question}"

        except Exception as e:
            logger.error(f"Synthesis call error on attempt {attempts}: {e}")
            circuit_breaker.record_failure()

    # If all healing attempts fail: Route to DLQ & serve fallback
    send_to_dlq(question, "Exceeded retry threshold / ungrounded answer", retrieved_docs, attempts)
    return QueryResult(
        answer="Autonomous retry threshold reached. Query logged to DLQ for diagnostic review.",
        status="EXHAUSTED_TO_DLQ",
        healed=healed,
        remedial_attempts=attempts,
        circuit_state=circuit_breaker.state.value,
        context_used=retrieved_docs
    )

@app.get("/api/v1/dlq")
def get_dead_letter_queue():
    """Retrieve logged failure records from Dead-Letter Queue."""
    return {"dlq_size": len(DEAD_LETTER_QUEUE), "records": DEAD_LETTER_QUEUE}
