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
    title="Day 40 Autonomous Multi-Agent Swarm RAG API",
    version="1.0.0",
    description="Role-specialized agent swarm with independent retrieval, conflict auditing, and consensus arbitration"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day40_multi_agent_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day40"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class SwarmRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class AgentEvaluation(BaseModel):
    role_name: str
    targeted_query: str
    retrieved_evidence: str
    domain_assessment: str

class SwarmResponse(BaseModel):
    session_id: str
    standalone_question: str
    agent_evaluations: List[AgentEvaluation]
    identified_conflicts: List[str]
    consensus_verdict: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 40 Multi-Agent Swarm RAG",
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

def retrieve_top_chunk(query: str, target_doc: str = "ALL") -> str:
    if collection.count() == 0:
        return "No indexed documentation available."

    q_vec = embed_model.encode(query).tolist()
    where_filter = {"source": target_doc} if target_doc != "ALL" else None
    n_res = min(4, collection.count())

    results = collection.query(query_embeddings=[q_vec], n_results=n_res, where=where_filter)
    if not results['documents'] or not results['documents'][0]:
        return "No relevant documentation found."

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    best_idx = int(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[0])

    m = metas[best_idx]
    return f"[{m.get('source')} Page {m.get('page')}]: {candidates[best_idx]}"

def run_specialist_agent(role: str, focus: str, question: str, target_doc: str) -> AgentEvaluation:
    targeted_query = f"{question} {focus}"
    evidence = retrieve_top_chunk(targeted_query, target_doc)

    prompt = f"""You are the {role} in an executive multi-agent swarm.
Your Area of Focus: {focus}

User Question: "{question}"
Retrieved Factual Evidence:
{evidence}

Provide a concise, domain-grounded assessment (2-3 sentences) evaluating this question strictly through the lens of your specialized role."""
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        assessment = res.choices[0].message.content.strip()
    except Exception:
        assessment = f"{role} assessment based on retrieved evidence: {evidence[:150]}"

    return AgentEvaluation(
        role_name=role,
        targeted_query=targeted_query,
        retrieved_evidence=evidence,
        domain_assessment=assessment
    )

def audit_conflicts_and_arbitrate(question: str, evals: List[AgentEvaluation]) -> tuple[List[str], str]:
    agent_reports = "\n\n".join([
        f"--- {e.role_name} Assessment ---\nFocus Query: {e.targeted_query}\nEvidence: {e.retrieved_evidence}\nAssessment: {e.domain_assessment}"
        for e in evals
    ])

    prompt = f"""You are the Executive Swarm Arbiter.
Review the independent evaluations provided by three specialized agents for the given question:

User Question: "{question}"

Specialist Agent Reports:
{agent_reports}

Tasks:
1. Identify any explicit or implicit trade-offs, disagreements, or risks between the agents' assessments.
2. Produce a balanced, authoritative final consensus verdict synthesizing all perspectives.

Respond STRICTLY with JSON:
{{
  "conflicts_or_tradeoffs": ["Conflict/tradeoff 1", "Conflict/tradeoff 2"],
  "consensus_verdict": "Complete, structured executive synthesis here..."
}}"""
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        raw = res.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        data = json.loads(raw.strip())
        return data.get("conflicts_or_tradeoffs", []), data.get("consensus_verdict", "Consensus reached.")
    except Exception:
        return ["Trade-offs identified across engineering feasibility and governance risk."], agent_reports

@app.post("/chat", response_model=SwarmResponse)
def swarm_chat(payload: SwarmRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="Document repository empty. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    # Run Specialist Swarm Agents
    agents_config = [
        ("Technical Architect Agent", "system architecture, infrastructure scalability, technical performance"),
        ("Compliance & Risk Agent", "regulatory security standards, privacy guidelines, policy adherence"),
        ("Operational & Cost Agent", "operational overhead, budget allocation, cost-benefit trade-offs")
    ]

    evaluations: List[AgentEvaluation] = []
    for role, focus in agents_config:
        agent_res = run_specialist_agent(role, focus, payload.question, payload.target_doc)
        evaluations.append(agent_res)

    # Cross-Agent Conflict Resolution & Consensus Arbitration
    conflicts, consensus_verdict = audit_conflicts_and_arbitrate(payload.question, evaluations)

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": consensus_verdict})

    return SwarmResponse(
        session_id=session_id,
        standalone_question=payload.question,
        agent_evaluations=evaluations,
        identified_conflicts=conflicts,
        consensus_verdict=consensus_verdict
    )
