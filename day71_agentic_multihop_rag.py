import time
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [MULTIHOP_AGENT] %(message)s")
logger = logging.getLogger("MultiHopAgent")

app = FastAPI(
    title="Day 71 Enterprise Agentic Multi-Hop RAG Engine",
    version="14.0.0",
    description="Multi-hop agentic reasoning with iterative context expansion and sufficiency evaluation"
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
collection = chroma_client.get_or_create_collection(name="day71_enterprise_docs")
groq_client = Groq(api_key="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")

# ----------------- Seed Dispersed Multi-Hop Knowledge Corpus -----------------
DISPERSED_CORPUS = [
    {
        "id": "doc_incident_alpha",
        "text": "Incident INC-9042 Report: The payment transaction service failed due to a severe bottleneck in the secondary storage tier called NebulaDB.",
        "category": "incidents"
    },
    {
        "id": "doc_nebuladb_specs",
        "text": "NebulaDB Architecture Spec: NebulaDB is configured with a strict connection limit of 450 concurrent sessions and max write throughput of 1,200 IOPS.",
        "category": "database_specs"
    },
    {
        "id": "doc_redis_specs",
        "text": "Redis Enterprise Cache Spec: The in-memory cache layer supports up to 85,000 IOPS and handles up to 10,000 active concurrent client sessions.",
        "category": "cache_specs"
    }
]

def seed_database():
    if collection.count() > 0:
        return
    docs = [item["text"] for item in DISPERSED_CORPUS]
    ids = [item["id"] for item in DISPERSED_CORPUS]
    metadatas = [{"category": item["category"]} for item in DISPERSED_CORPUS]
    collection.add(documents=docs, ids=ids, metadatas=metadatas)
    logger.info(f"Seeded {len(docs)} multi-hop documents into ChromaDB.")

seed_database()

# ----------------- Agent State Representation -----------------
class AgentState(BaseModel):
    original_query: str
    current_sub_query: str
    accumulated_context: List[str] = Field(default_factory=list)
    hops_completed: int = 0
    max_hops: int = 3
    is_sufficient: bool = False
    hop_history: List[Dict[str, Any]] = Field(default_factory=list)

# ----------------- Agent Nodes -----------------
def retrieve_node(query: str, top_k: int = 1) -> List[Dict[str, Any]]:
    q_vec = embed_model.encode(query).tolist()
    res = collection.query(query_embeddings=[q_vec], n_results=top_k)
    results = []
    if res["ids"] and res["ids"][0]:
        for doc_id, text in zip(res["ids"][0], res["documents"][0]):
            results.append({"id": doc_id, "text": text})
    return results

def evaluate_sufficiency_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates if accumulated evidence is sufficient to answer the prompt.
    If not, it outputs the next targeted sub-query to investigate.
    """
    context_str = "\n".join([f"- {c}" for c in state.accumulated_context])
    
    prompt = f"""You are a multi-hop RAG orchestration agent.
Original User Query: "{state.original_query}"
Current Accumulated Context:
{context_str}

Evaluate if the accumulated context contains enough information to fully and completely answer the original query.
Output a strict JSON object with two fields:
1. "is_sufficient": boolean (true if all facts needed to answer the query are present, false otherwise)
2. "next_sub_query": string (if false, specify the exact targeted search query to find the missing fact. If true, set to "")

Raw JSON only:"""

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150
        )
        content = completion.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)
    except Exception as e:
        logger.warning(f"Evaluation fallback triggered: {e}")
        # Rule-based fallback for the multi-hop sequence
        if "NebulaDB" in context_str and not any("1,200 IOPS" in c or "1200 IOPS" in c for c in state.accumulated_context):
            return {"is_sufficient": False, "next_sub_query": "What is the max write throughput of NebulaDB?"}
        return {"is_sufficient": True, "next_sub_query": ""}

# ----------------- Request / Response Models -----------------
class MultiHopQueryRequest(BaseModel):
    query: str = Field(..., example="What is the maximum write throughput of the storage engine that failed in incident INC-9042?")
    max_hops: int = Field(default=3, ge=1, le=5)

class MultiHopQueryResponse(BaseModel):
    original_query: str
    final_answer: str
    total_hops_executed: int
    hops_detail: List[Dict[str, Any]]

# ----------------- API Endpoints -----------------
@app.post("/api/v1/multihop-query", response_model=MultiHopQueryResponse)
async def execute_multihop_rag(payload: MultiHopQueryRequest):
    q_str = payload.query.strip()
    if not q_str:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Initialize Graph State
    state = AgentState(
        original_query=q_str,
        current_sub_query=q_str,
        max_hops=payload.max_hops
    )

    # Iterative Agent Loop
    while state.hops_completed < state.max_hops:
        state.hops_completed += 1
        
        # Step 1: Retrieval Node
        retrieved_chunks = retrieve_node(state.current_sub_query, top_k=1)
        step_texts = [item["text"] for item in retrieved_chunks]
        
        # Add non-duplicate chunks to accumulated context
        for t in step_texts:
            if t not in state.accumulated_context:
                state.accumulated_context.append(t)

        # Log Hop Trace
        state.hop_history.append({
            "hop": state.hops_completed,
            "searched_query": state.current_sub_query,
            "retrieved_docs": [item["id"] for item in retrieved_chunks],
            "evidence": step_texts
        })

        # Step 2: Evaluation & Routing Node
        eval_result = evaluate_sufficiency_node(state)
        state.is_sufficient = eval_result.get("is_sufficient", False)
        
        if state.is_sufficient:
            logger.info(f"Context sufficiency achieved at hop {state.hops_completed}.")
            break
        
        # Update sub-query for the next hop
        next_query = eval_result.get("next_sub_query", "").strip()
        if not next_query or next_query == state.current_sub_query:
            break
        state.current_sub_query = next_query

    # Step 3: Synthesis Node
    context_str = "\n".join([f"- {c}" for c in state.accumulated_context])
    synthesis_prompt = f"""Context gathered across multi-hop investigations:
{context_str}

Question: {state.original_query}
Provide a direct, factual answer stating the exact technical specs requested in 1-2 clear sentences:"""

    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": synthesis_prompt}],
        temperature=0.1,
        max_tokens=150
    )
    final_answer = resp.choices[0].message.content

    return MultiHopQueryResponse(
        original_query=state.original_query,
        final_answer=final_answer,
        total_hops_executed=state.hops_completed,
        hops_detail=state.hop_history
    )
