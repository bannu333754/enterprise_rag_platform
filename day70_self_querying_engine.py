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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [SELF_QUERY] %(message)s")
logger = logging.getLogger("SelfQueryingEngine")

app = FastAPI(
    title="Day 70 Enterprise Self-Querying Metadata Engine",
    version="13.0.0",
    description="Query construction decomposing natural language into semantic vectors and structured metadata filters"
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
collection = chroma_client.get_or_create_collection(name="day70_incidents_kb")
groq_client = Groq(api_key="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")

# ----------------- Document Corpus with Structured Metadata -----------------
INCIDENTS_METADATA_CORPUS = [
    {
        "id": "inc_2026_eu_p1",
        "text": "Outage report: EU Frankfurt cluster database connection pool exhausted due to unindexed query spikes during peak traffic.",
        "metadata": {"year": 2026, "severity": "P1", "region": "EU", "service": "database"}
    },
    {
        "id": "inc_2025_eu_p1",
        "text": "Historical incident: EU Ireland datacenter power surge caused automated failover of secondary Redis shards.",
        "metadata": {"year": 2025, "severity": "P1", "region": "EU", "service": "redis"}
    },
    {
        "id": "inc_2026_us_p1",
        "text": "US East ingress controller CPU throttling dropped inbound API gateway requests under sustained DDoS attacks.",
        "metadata": {"year": 2026, "severity": "P1", "region": "US", "service": "gateway"}
    },
    {
        "id": "inc_2026_eu_p3",
        "text": "EU cluster staging environment experienced minor telemetry dashboard latency spikes during scheduled maintenance.",
        "metadata": {"year": 2026, "severity": "P3", "region": "EU", "service": "observability"}
    }
]

def seed_database():
    if collection.count() > 0:
        return

    docs = [item["text"] for item in INCIDENTS_METADATA_CORPUS]
    ids = [item["id"] for item in INCIDENTS_METADATA_CORPUS]
    metadatas = [item["metadata"] for item in INCIDENTS_METADATA_CORPUS]

    collection.add(documents=docs, ids=ids, metadatas=metadatas)
    logger.info(f"Seeded {len(docs)} incident documents with structured metadata.")

seed_database()

# ----------------- Query Construction Parser -----------------
class ParsedQuerySchema(BaseModel):
    semantic_query: str = Field(..., description="The thematic search phrase with metadata values stripped away")
    year: Optional[int] = Field(None, description="Extracted year integer (e.g. 2025, 2026) or null")
    severity: Optional[str] = Field(None, description="Extracted severity level: P1, P2, P3, or null")
    region: Optional[str] = Field(None, description="Extracted region: EU, US, APAC, or null")
    service: Optional[str] = Field(None, description="Extracted service: database, redis, gateway, observability, or null")

def construct_structured_query(user_prompt: str) -> ParsedQuerySchema:
    """
    Prompts Groq LLM to decompose a user query into semantic search text and discrete metadata filters.
    """
    system_prompt = (
        "You are an enterprise query construction parser. Your task is to extract:\n"
        "1. semantic_query: the conceptual intent without filter words.\n"
        "2. Metadata attributes: year (integer), severity (P1/P2/P3), region (EU/US/APAC), service (database/redis/gateway/observability).\n"
        "Output ONLY a raw JSON object matching the requested schema with no markdown formatting."
    )

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Deconstruct this query: \"{user_prompt}\""}
            ],
            temperature=0.0,
            max_tokens=150
        )
        content = completion.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)
        return ParsedQuerySchema(**data)
    except Exception as e:
        logger.warning(f"Structured parser fallback triggered: {e}")
        # Rule-based fallback extraction
        year = 2026 if "2026" in user_prompt else (2025 if "2025" in user_prompt else None)
        severity = "P1" if "p1" in user_prompt.lower() else ("P3" if "p3" in user_prompt.lower() else None)
        region = "EU" if "eu" in user_prompt.lower() else ("US" if "us" in user_prompt.lower() else None)
        clean_text = user_prompt.replace("2026", "").replace("2025", "").replace("P1", "").replace("p1", "").replace("EU", "").strip()
        return ParsedQuerySchema(semantic_query=clean_text or user_prompt, year=year, severity=severity, region=region)

# ----------------- Request / Response Models -----------------
class SelfQueryRequest(BaseModel):
    query: str = Field(..., example="Find 2026 P1 database incidents in the EU region")

class RetrievedIncident(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]

class SelfQueryResponse(BaseModel):
    original_query: str
    parsed_semantic_query: str
    extracted_filters: Dict[str, Any]
    retrieved_incidents: List[RetrievedIncident]
    answer: str

# ----------------- API Endpoints -----------------
@app.post("/api/v1/self-query", response_model=SelfQueryResponse)
async def execute_self_query(payload: SelfQueryRequest):
    q_str = payload.query.strip()
    if not q_str:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Parse structured metadata and isolated semantic query
    parsed = construct_structured_query(q_str)

    # 2. Build ChromaDB native boolean filter
    filter_conditions = []
    if parsed.year is not None:
        filter_conditions.append({"year": {"$eq": parsed.year}})
    if parsed.severity is not None:
        filter_conditions.append({"severity": {"$eq": parsed.severity}})
    if parsed.region is not None:
        filter_conditions.append({"region": {"$eq": parsed.region}})
    if parsed.service is not None:
        filter_conditions.append({"service": {"$eq": parsed.service}})

    where_clause = None
    if len(filter_conditions) == 1:
        where_clause = filter_conditions[0]
    elif len(filter_conditions) > 1:
        where_clause = {"$and": filter_conditions}

    # 3. Vector Search using isolated semantic query
    q_vec = embed_model.encode(parsed.semantic_query).tolist()
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=3,
        where=where_clause
    )

    retrieved_items: List[RetrievedIncident] = []
    snippets = []

    if results["ids"] and results["ids"][0]:
        for doc_id, text, meta in zip(results["ids"][0], results["documents"][0], results["metadatas"][0]):
            retrieved_items.append(RetrievedIncident(id=doc_id, text=text, metadata=meta))
            snippets.append(f"[{doc_id}]: {text}")

    context_str = "\n".join(snippets) if snippets else "No matching incident records found."

    # 4. LLM Synthesis
    prompt = (
        f"Incident Context:\n{context_str}\n\n"
        f"Query: {q_str}\n"
        f"Synthesize an incident status report in 1-2 factual sentences:"
    )

    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=150
    )
    answer = resp.choices[0].message.content

    return SelfQueryResponse(
        original_query=q_str,
        parsed_semantic_query=parsed.semantic_query,
        extracted_filters=where_clause or {},
        retrieved_incidents=retrieved_items,
        answer=answer
    )
