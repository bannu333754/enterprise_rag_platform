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
    title="Day 48 Contrastive HyDE Multi-Hypothesis RAG API",
    version="1.0.0",
    description="Dual-hypothesis generation, contrastive retrieval fusion, and multi-perspective synthesis"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day48_contrastive_hyde_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day48"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class ContrastiveHyDERequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"
    contrast_mode: str = "POLARIZED"  # "POLARIZED" or "COMPLEMENTARY"

class HypothesisDetail(BaseModel):
    hypothesis_id: str
    label: str
    synthetic_passage: str

class ContrastiveCandidate(BaseModel):
    source: str
    page: int
    matched_via: str
    cross_score: float
    snippet: str

class ContrastiveHyDEResponse(BaseModel):
    session_id: str
    standalone_question: str
    hypotheses: List[HypothesisDetail]
    retrieved_passages: List[ContrastiveCandidate]
    synthesis_verdict: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 48 Contrastive HyDE RAG",
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

def generate_contrastive_hypotheses(question: str, mode: str = "POLARIZED") -> List[HypothesisDetail]:
    """Generates two distinct, contrasting hypothetical document passages for the query."""
    prompt = f"""You are a Contrastive Hypothetical Document Generator for an advanced RAG engine.
Given the question, write TWO distinct, technically sound hypothetical answers that an authoritative document might state.

Mode: {mode}
- If POLARIZED: Hypothesis A highlights the primary/positive perspective or standard consensus; Hypothesis B highlights alternative paradigms, trade-offs, edge-case limitations, or opposing mechanisms.
- If COMPLEMENTARY: Hypothesis A focuses on structural/architectural details; Hypothesis B focuses on operational/performance metrics.

Question: "{question}"

Respond STRICTLY with JSON:
{{
  "hypotheses": [
    {{
      "hypothesis_id": "H1",
      "label": "Perspective A / Standard Focus",
      "synthetic_passage": "Concise paragraph detailing answer perspective A..."
    }},
    {{
      "hypothesis_id": "H2",
      "label": "Perspective B / Alternative or Trade-off Focus",
      "synthetic_passage": "Concise paragraph detailing answer perspective B..."
    }}
  ]
}}"""
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        raw = res.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        data = json.loads(raw.strip())
        hypos = []
        for h in data.get("hypotheses", []):
            hypos.append(HypothesisDetail(
                hypothesis_id=h.get("hypothesis_id", "H"),
                label=h.get("label", "Hypothesis"),
                synthetic_passage=h.get("synthetic_passage", "")
            ))
        return hypos if hypos else [HypothesisDetail(hypothesis_id="H1", label="Default", synthetic_passage=question)]
    except Exception:
        return [
            HypothesisDetail(hypothesis_id="H1", label="Standard Projection", synthetic_passage=f"Primary considerations for {question}"),
            HypothesisDetail(hypothesis_id="H2", label="Alternative Projection", synthetic_passage=f"Risk and operational caveats for {question}")
        ]

def contrastive_retrieve(
    question: str,
    hypotheses: List[HypothesisDetail],
    target_doc: str = "ALL"
) -> List[ContrastiveCandidate]:
    """Executes multi-hypothesis contrastive vector retrieval, merges, and cross-ranks."""
    if collection.count() == 0:
        return []

    where_filter = {"source": target_doc} if target_doc != "ALL" else None
    seen_chunks: Dict[str, dict] = {}

    # Query 1: Direct question vector
    queries_to_run = [("Direct Query", question)]
    for h in hypotheses:
        queries_to_run.append((h.label, h.synthetic_passage))

    for source_label, query_text in queries_to_run:
        q_vec = embed_model.encode(query_text).tolist()
        n_res = min(4, collection.count())
        res = collection.query(query_embeddings=[q_vec], n_results=n_res, where=where_filter)

        if res['documents'] and res['documents'][0]:
            for doc, meta in zip(res['documents'][0], res['metadatas'][0]):
                key = f"{meta.get('source')}_p{meta.get('page')}_{doc[:40]}"
                if key not in seen_chunks:
                    seen_chunks[key] = {
                        "text": doc,
                        "meta": meta,
                        "matched_via": source_label
                    }

    candidates = list(seen_chunks.values())
    if not candidates:
        return []

    # Cross-Encoder re-ranking against the core user question
    pairs = [[question, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:4]

    results: List[ContrastiveCandidate] = []
    for idx in ranked_indices:
        item = candidates[idx]
        sc = round(float(scores[idx]), 3)
        m = item["meta"]
        results.append(ContrastiveCandidate(
            source=m.get("source", "Unknown"),
            page=m.get("page", 0),
            matched_via=item["matched_via"],
            cross_score=sc,
            snippet=item["text"]
        ))
    return results

@app.post("/chat", response_model=ContrastiveHyDEResponse)
def contrastive_hyde_chat(payload: ContrastiveHyDERequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="Vector index is empty. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    # Step 1: Multi-Hypothesis Contrastive Generation
    hypotheses = generate_contrastive_hypotheses(payload.question, payload.contrast_mode)

    # Step 2: Multi-Vector Contrastive Retrieval & Reranking
    retrieved_candidates = contrastive_retrieve(payload.question, hypotheses, payload.target_doc)

    evidence_text = "\n\n".join([
        f"[{c.source} Page {c.page} | Retrieved via {c.matched_via} | Score: {c.cross_score}]:\n{c.snippet}"
        for c in retrieved_candidates
    ]) if retrieved_candidates else "No direct matching passages retrieved."

    # Step 3: Contrastive Multi-Perspective Synthesis
    system_prompt = (
        "You are an enterprise AI utilizing Contrastive Multi-Hypothesis RAG (Contrastive HyDE). "
        "Synthesize a balanced, highly nuanced response. Compare and integrate the evidence surfaced "
        "across the contrasting perspectives, highlighting trade-offs, edge conditions, and definitive factual findings."
    )
    user_msg = f"--- SURFACED CONTRASTIVE EVIDENCE ---\n{evidence_text}\n\n--- QUESTION ---\n{payload.question}"

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_msg})

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.2
        )
        final_answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": final_answer})

    return ContrastiveHyDEResponse(
        session_id=session_id,
        standalone_question=payload.question,
        hypotheses=hypotheses,
        retrieved_passages=retrieved_candidates,
        synthesis_verdict=final_answer
    )
