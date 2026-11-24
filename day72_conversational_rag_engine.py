import time
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [CONVERSATIONAL_RAG] %(message)s")
logger = logging.getLogger("ConversationalRAG")

app = FastAPI(
    title="Day 72 Enterprise Conversational RAG & Session Memory Engine",
    version="15.0.0",
    description="Multi-turn contextual RAG with standalone query condensation and rolling session memory"
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
collection = chroma_client.get_or_create_collection(name="day72_enterprise_infrastructure")
groq_client = Groq(api_key="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")

SESSION_STORAGE: Dict[str, List[Dict[str, str]]] = {}

INFRA_CORPUS = [
    {
        "id": "doc_envoy_gateway",
        "text": "Envoy Gateway acts as the enterprise Kubernetes ingress controller. It listens natively on ingress port 8443 for mutual TLS (mTLS) traffic and routes internal gRPC calls on port 9090."
    },
    {
        "id": "doc_envoy_rate_limit",
        "text": "Envoy Gateway integrates an external global rate-limit filter service. The default bucket limit is configured to 2,500 requests per minute per IP address before issuing HTTP 429."
    },
    {
        "id": "doc_clickhouse_specs",
        "text": "ClickHouse is deployed for cold telemetry analytics. It consumes log streams directly from Kafka topics and stores aggregated parquet partitions across AWS S3 object stores."
    }
]

def seed_database():
    if collection.count() > 0:
        return
    docs = [item["text"] for item in INFRA_CORPUS]
    ids = [item["id"] for item in INFRA_CORPUS]
    collection.add(documents=docs, ids=ids)
    logger.info(f"Seeded {len(docs)} infrastructure knowledge documents.")

seed_database()

def condense_standalone_query(chat_history: List[Dict[str, str]], latest_question: str) -> str:
    if not chat_history:
        return latest_question

    formatted_history = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history[-4:]])
    prompt = (
        f"Conversation History:\n{formatted_history}\n\n"
        f"Latest Question: {latest_question}\n\n"
        "Task: Rewrite the latest question into a single standalone query that explicitly mentions the specific system or entity being discussed (e.g. 'Envoy Gateway'). "
        "Do not answer the question. Only output the rephrased standalone query text."
    )

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60
        )
        condensed = completion.choices[0].message.content.strip().replace('"', '').replace("Standalone Query:", "").strip()
        if condensed and len(condensed) > 5 and ("envoy" in condensed.lower() or "clickhouse" in condensed.lower()):
            return condensed
    except Exception as e:
        logger.warning(f"Query condensation exception: {e}")

    # Fallback entity resolution
    history_str = " ".join([m["content"] for m in chat_history]).lower()
    subject = "Envoy Gateway" if "envoy" in history_str else ("ClickHouse" if "clickhouse" in history_str else "")
    if subject and any(p in latest_question.lower() for p in ["it", "its", "this"]):
        return f"{latest_question} ({subject})"
    return latest_question

class ConversationalQueryRequest(BaseModel):
    session_id: str = Field(..., example="session_user_402")
    question: str = Field(..., example="What port does it listen on for mTLS?")

class ConversationalQueryResponse(BaseModel):
    session_id: str
    original_question: str
    condensed_query: str
    retrieved_chunk_id: str
    answer: str
    history_depth: int

@app.post("/api/v1/chat", response_model=ConversationalQueryResponse)
async def chat_interaction(payload: ConversationalQueryRequest):
    sess_id = payload.session_id.strip()
    raw_q = payload.question.strip()

    if not sess_id or not raw_q:
        raise HTTPException(status_code=400, detail="session_id and question cannot be empty.")

    if sess_id not in SESSION_STORAGE:
        SESSION_STORAGE[sess_id] = []
    
    current_history = SESSION_STORAGE[sess_id]

    standalone_query = condense_standalone_query(current_history, raw_q)
    logger.info(f"[{sess_id}] Raw: '{raw_q}' -> Condensed: '{standalone_query}'")

    q_vec = embed_model.encode(standalone_query).tolist()
    search_res = collection.query(query_embeddings=[q_vec], n_results=1)

    retrieved_text = search_res["documents"][0][0] if search_res["documents"] and search_res["documents"][0] else "No knowledge available."
    retrieved_id = search_res["ids"][0][0] if search_res["ids"] and search_res["ids"][0] else "none"

    synthesis_prompt = (
        f"Context Document:\n{retrieved_text}\n\n"
        f"Question: {standalone_query}\n"
        f"Provide a direct, factual answer stating the exact technical details in 1 sentence:"
    )

    try:
        resp = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.1,
            max_tokens=120
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        answer = retrieved_text

    if not answer:
        answer = retrieved_text

    current_history.append({"role": "user", "content": raw_q})
    current_history.append({"role": "assistant", "content": answer})

    if len(current_history) > 10:
        SESSION_STORAGE[sess_id] = current_history[-10:]

    return ConversationalQueryResponse(
        session_id=sess_id,
        original_question=raw_q,
        condensed_query=standalone_query,
        retrieved_chunk_id=retrieved_id,
        answer=answer,
        history_depth=len(SESSION_STORAGE[sess_id])
    )

@app.delete("/api/v1/chat/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in SESSION_STORAGE:
        del SESSION_STORAGE[session_id]
        return {"status": "success", "message": f"Session {session_id} evicted."}
    return {"status": "not_found", "message": "Session does not exist."}
