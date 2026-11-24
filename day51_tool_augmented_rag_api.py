import os
import json
import sqlite3
import shutil
import chromadb
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(
    title="Day 51 Tool-Augmented Schema-Enforced RAG API",
    version="1.0.0",
    description="Deterministic function calling, tool execution loops, and schema-constrained extraction"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day51_tool_augmented_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day51"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory SQLite Database for Structured Query Execution
conn = sqlite3.connect(":memory:", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE enterprise_metrics (
        metric_name TEXT,
        department TEXT,
        current_value REAL,
        threshold_limit REAL,
        status TEXT
    )
""")
cursor.executemany("""
    INSERT INTO enterprise_metrics VALUES (?, ?, ?, ?, ?)
""", [
    ("api_latency_p99", "Core Backend", 420.5, 300.0, "DEGRADED"),
    ("memory_utilization", "Cluster Infra", 68.2, 85.0, "HEALTHY"),
    ("failed_jobs_count", "Data Engineering", 14.0, 5.0, "CRITICAL"),
    ("active_worker_threads", "Ingestion Service", 128.0, 256.0, "HEALTHY")
])
conn.commit()

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

# Strict Schema Models
class ActionPayload(BaseModel):
    action_type: str = Field(..., description="Action: 'DISPATCH_ALERT', 'EXECUTE_QUERY', or 'NO_ACTION'")
    target_entity: str
    severity: str = Field(default="INFO", description="'INFO', 'WARNING', 'CRITICAL'")
    parameters: Dict[str, Any]

class StructuredExtractionSchema(BaseModel):
    summary: str
    identified_issues: List[str]
    action_required: bool
    recommended_action: Optional[ActionPayload] = None

class ToolExecutionResult(BaseModel):
    tool_name: str
    parameters_passed: Dict[str, Any]
    execution_status: str
    output_data: Any

class ToolAugmentedRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class ToolAugmentedResponse(BaseModel):
    session_id: str
    standalone_question: str
    structured_data: StructuredExtractionSchema
    executed_tools: List[ToolExecutionResult]
    final_conversational_answer: str

# Local Deterministic Tools
def tool_execute_sql(query_pattern: str) -> List[Dict[str, Any]]:
    """Safe read-only execution against internal enterprise metrics table."""
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM enterprise_metrics")
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in rows]
    except Exception as e:
        return [{"error": str(e)}]

def tool_dispatch_alert(target_entity: str, severity: str, details: str) -> Dict[str, Any]:
    """Simulates an enterprise alert dispatch to Slack/PagerDuty."""
    return {
        "status": "DISPATCHED",
        "service": target_entity,
        "severity": severity,
        "audit_code": f"ALERT-SEC-{severity[:3]}-902",
        "message": details
    }

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 51 Tool-Augmented Schema-Enforced RAG",
        "total_chunks": collection.count(),
        "active_sessions": len(SESSION_HISTORIES)
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    reader = PdfReader(file_path)
    total_added = 0

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        words = text.split()
        chunk_size = 75
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        for c_idx, chunk in enumerate(chunks):
            doc_id = f"{file.filename}_p{page_idx + 1}_c{c_idx}"
            vector = embed_model.encode(chunk).tolist()

            collection.add(
                documents=[chunk],
                embeddings=[vector],
                metadatas=[{"source": file.filename, "page": page_idx + 1}],
                ids=[doc_id]
            )
            total_added += 1

    return {
        "status": "Success",
        "filename": file.filename,
        "chunks_indexed": total_added,
        "total_chunks": collection.count()
    }

@app.post("/chat", response_model=ToolAugmentedResponse)
def tool_augmented_chat(payload: ToolAugmentedRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    context_str = "No documents indexed."
    if collection.count() > 0:
        q_vec = embed_model.encode(payload.question).tolist()
        where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None
        n_res = min(4, collection.count())

        results = collection.query(query_embeddings=[q_vec], n_results=n_res, where=where_filter)
        if results['documents'] and results['documents'][0]:
            candidates = results['documents'][0]
            metas = results['metadatas'][0]

            pairs = [[payload.question, doc] for doc in candidates]
            scores = reranker.predict(pairs)
            best_idx = int(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[0])
            m = metas[best_idx]
            context_str = f"[{m.get('source')} Page {m.get('page')}]: {candidates[best_idx]}"

    # Step 1: Schema-Enforced Extraction & Action Planning
    schema_prompt = f"""You are an Enterprise Schema Extraction & Tool Orchestration Engine.
Analyze the user question and retrieved evidence. Output a STRICT JSON matching this schema:

{{
  "summary": "Concise summary of findings",
  "identified_issues": ["list of issues or observations"],
  "action_required": true or false,
  "recommended_action": {{
    "action_type": "DISPATCH_ALERT" or "EXECUTE_QUERY" or "NO_ACTION",
    "target_entity": "Name of system or metric",
    "severity": "INFO" or "WARNING" or "CRITICAL",
    "parameters": {{"key": "value"}}
  }}
}}

Context:
{context_str}

User Question:
{payload.question}"""

    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": schema_prompt}],
            temperature=0.0
        )
        raw = res.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        structured_dict = json.loads(raw.strip())
        structured_obj = StructuredExtractionSchema(**structured_dict)
    except Exception:
        structured_obj = StructuredExtractionSchema(
            summary="Processed query via fallback parsing.",
            identified_issues=["Standard factual lookup"],
            action_required=False,
            recommended_action=None
        )

    # Step 2: Dynamic Tool Execution Loop
    executed_tools_list: List[ToolExecutionResult] = []
    if structured_obj.action_required and structured_obj.recommended_action:
        act = structured_obj.recommended_action
        if act.action_type == "EXECUTE_QUERY":
            sql_out = tool_execute_sql(act.target_entity)
            executed_tools_list.append(ToolExecutionResult(
                tool_name="tool_execute_sql",
                parameters_passed=act.parameters,
                execution_status="SUCCESS",
                output_data=sql_out
            ))
        elif act.action_type == "DISPATCH_ALERT":
            alert_out = tool_dispatch_alert(act.target_entity, act.severity, structured_obj.summary)
            executed_tools_list.append(ToolExecutionResult(
                tool_name="tool_dispatch_alert",
                parameters_passed=act.parameters,
                execution_status="DISPATCHED",
                output_data=alert_out
            ))

    # Step 3: Synthesis with Integrated Tool Results
    tool_audit_text = json.dumps([t.dict() for t in executed_tools_list], indent=2) if executed_tools_list else "None"
    synthesis_prompt = f"""You are an authoritative Enterprise AI. 
Synthesize a professional response for the user incorporating the context evidence and any executed tool actions.

Context Evidence:
{context_str}

Executed Action Outputs:
{tool_audit_text}

User Question:
{payload.question}"""

    try:
        final_res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.2
        )
        final_answer = final_res.choices[0].message.content.strip()
    except Exception as e:
        final_answer = f"Error during final synthesis: {str(e)}"

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": final_answer})

    return ToolAugmentedResponse(
        session_id=session_id,
        standalone_question=payload.question,
        structured_data=structured_obj,
        executed_tools=executed_tools_list,
        final_conversational_answer=final_answer
    )
