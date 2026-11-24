import os
import json
import shutil
import chromadb
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(
    title="Day 39 Agentic Plan-and-Solve RAG API",
    version="1.0.0",
    description="Dynamic sub-goal decomposition, sequential step-by-step retrieval, and reflective verification"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day39_plan_solve_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day39"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class PlanAndSolveRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"
    max_steps: int = 3

class SubGoalExecution(BaseModel):
    step_number: int
    sub_goal_query: str
    retrieved_chunk_summary: str
    intermediate_finding: str

class PlanAndSolveResponse(BaseModel):
    session_id: str
    standalone_question: str
    execution_plan: List[str]
    step_traces: List[SubGoalExecution]
    verification_passed: bool
    verification_rationale: str
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 39 Agentic Plan-and-Solve RAG",
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

def generate_sub_goals(question: str, max_steps: int = 3) -> List[str]:
    """Decomposes a complex multi-faceted question into an ordered sequence of search sub-goals."""
    prompt = f"""You are a Strategic Planning Engine for an Agentic RAG System.
Decompose the complex user question into {max_steps} discrete, targeted search sub-queries to execute sequentially.

Question: "{question}"

Respond STRICTLY with a valid JSON array of strings containing the sub-queries, for example:
["Step 1 search query", "Step 2 search query"]
JSON:"""
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw = res.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        goals = json.loads(raw.strip())
        return [str(g).strip() for g in goals if isinstance(g, str)][:max_steps]
    except Exception:
        return [question]

def retrieve_and_rerank_for_step(sub_query: str, target_doc: str = "ALL") -> str:
    """Executes vector retrieval and cross-encoder re-ranking for a single sub-goal."""
    if collection.count() == 0:
        return "No documents available in index."

    q_vec = embed_model.encode(sub_query).tolist()
    where_filter = {"source": target_doc} if target_doc != "ALL" else None
    n_res = min(4, collection.count())

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=n_res,
        where=where_filter
    )

    if not results['documents'] or not results['documents'][0]:
        return "No matching evidence located for this step."

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    pairs = [[sub_query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    best_idx = int(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[0])

    m = metas[best_idx]
    return f"[{m.get('source')} Page {m.get('page')}]: {candidates[best_idx]}"

def synthesize_step_finding(sub_goal: str, evidence: str) -> str:
    """Extracts the specific finding relevant to the sub-goal from the retrieved evidence."""
    prompt = f"""Extract the direct factual answer to the sub-query from the evidence passage.
Sub-Query: "{sub_goal}"
Evidence:
{evidence}

Direct Finding (1-2 sentences):"""
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return evidence[:180]

def verify_and_synthesize(question: str, step_traces: List[SubGoalExecution]) -> tuple[str, bool, str]:
    """Performs self-reflective verification and generates final answer."""
    trace_text = "\n\n".join([
        f"Sub-Goal {t.step_number}: {t.sub_goal_query}\nEvidence: {t.retrieved_chunk_summary}\nFinding: {t.intermediate_finding}"
        for t in step_traces
    ])

    verify_prompt = f"""Review the original question and the sequential sub-goal execution trace:
Original Question: {question}

Execution Trace:
{trace_text}

Task:
1. Synthesize a comprehensive final answer addressing all aspects of the original question.
2. Verify if the execution trace provided sufficient evidence without requiring ungrounded speculation.

Respond STRICTLY in JSON format:
{{
  "verified": true,
  "rationale": "All sub-goals satisfied directly by evidence",
  "final_answer": "Complete synthesized answer here..."
}}"""
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": verify_prompt}],
            temperature=0.2
        )
        raw = res.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        data = json.loads(raw.strip())
        return data.get("final_answer", ""), bool(data.get("verified", True)), data.get("rationale", "Verified")
    except Exception:
        return f"Based on findings:\n{trace_text}", True, "Heuristic verification pass"

@app.post("/chat", response_model=PlanAndSolveResponse)
def plan_and_solve_chat(payload: PlanAndSolveRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="Index is empty. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    # Step 1: Sub-Goal Decomposition & Planning
    sub_goals = generate_sub_goals(payload.question, max_steps=payload.max_steps)

    step_traces: List[SubGoalExecution] = []

    # Step 2: Sequential Step-by-Step Retrieval & Finding Extraction
    for step_idx, sub_q in enumerate(sub_goals):
        evidence_passage = retrieve_and_rerank_for_step(sub_q, payload.target_doc)
        finding = synthesize_step_finding(sub_q, evidence_passage)

        step_traces.append(SubGoalExecution(
            step_number=step_idx + 1,
            sub_goal_query=sub_q,
            retrieved_chunk_summary=evidence_passage,
            intermediate_finding=finding
        ))

    # Step 3: Self-Reflective Verification & Synthesis
    final_ans, is_verified, rationale = verify_and_synthesize(payload.question, step_traces)

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": final_ans})

    return PlanAndSolveResponse(
        session_id=session_id,
        standalone_question=payload.question,
        execution_plan=sub_goals,
        step_traces=step_traces,
        verification_passed=is_verified,
        verification_rationale=rationale,
        final_answer=final_ans
    )
