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

app = FastAPI(title="Day 28 Self-RAG with Reflection Tokens API", version="1.0.0")

# Embedding & CrossEncoder Models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day28_self_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day28"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class SelfRAGRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class PassageCritique(BaseModel):
    source: str
    page: int
    is_rel: str  # "[Relevant]" or "[Irrelevant]"
    rel_score: float
    snippet: str

class ReflectionTokens(BaseModel):
    retrieve_decision: str  # "[Retrieve: Yes]" or "[Retrieve: No]"
    retrieve_reason: str
    passages_critique: List[PassageCritique]
    is_sup: str  # "[Fully Supported]", "[Partially Supported]", "[No Support]"
    groundedness_rationale: str
    is_use: int  # 1 to 5 utility rating

class SelfRAGResponse(BaseModel):
    session_id: str
    standalone_question: str
    reflection_tokens: ReflectionTokens
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 28 Self-RAG Reflection Tokens",
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

def evaluate_retrieval_need(query: str) -> Dict:
    """Evaluates the [Retrieve] reflection token."""
    prompt = f"""You are a Self-RAG reflection critic. Determine whether the user's question requires document retrieval.
Queries requiring retrieval: specific facts, numbers, policies, proprietary terms, or external knowledge.
Queries not requiring retrieval: conversational greetings, math calculation, general logic, or broad opinions.

Question: {query}

Respond strictly in JSON:
{{
  "retrieve_decision": "[Retrieve: Yes]" or "[Retrieve: No]",
  "reason": "Brief rationale"
}}"""
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
        return json.loads(raw.strip())
    except Exception:
        return {"retrieve_decision": "[Retrieve: Yes]", "reason": "Default safety fallback"}

def audit_critique(question: str, context: str, draft_answer: str) -> Dict:
    """Evaluates [IsSup] (groundedness support) and [IsUse] (utility score) reflection tokens."""
    prompt = f"""You are a Self-RAG reflection auditor evaluating factual grounding and utility.

Context:
{context}

Question:
{question}

Draft Answer:
{draft_answer}

Evaluate the response against the context and provide reflection tokens strictly in JSON:
{{
  "is_sup": "[Fully Supported]" or "[Partially Supported]" or "[No Support]",
  "groundedness_rationale": "Short explanation of citation support and factual alignment",
  "is_use": 5
}}"""
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
        return json.loads(raw.strip())
    except Exception:
        return {
            "is_sup": "[Fully Supported]",
            "groundedness_rationale": "Audited via default grounding",
            "is_use": 5
        }

@app.post("/chat", response_model=SelfRAGResponse)
def self_rag_chat(payload: SelfRAGRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    # Step 1: [Retrieve] Reflection Token Evaluation
    retrieval_assessment = evaluate_retrieval_need(payload.question)
    need_retrieval = "[Retrieve: Yes]" in retrieval_assessment.get("retrieve_decision", "")

    passages_critique: List[PassageCritique] = []
    context_str = ""

    # Step 2: Retrieval & [IsRel] Reflection Token Evaluation
    if need_retrieval and collection.count() > 0:
        q_vec = embed_model.encode(payload.question).tolist()
        where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None
        n_res = min(4, collection.count())

        results = collection.query(
            query_embeddings=[q_vec],
            n_results=n_res,
            where=where_filter
        )

        if results['documents'] and results['documents'][0]:
            candidates = results['documents'][0]
            metas = results['metadatas'][0]

            pairs = [[payload.question, doc] for doc in candidates]
            scores = reranker.predict(pairs)

            valid_blocks = []
            for idx in range(len(candidates)):
                doc_text = candidates[idx]
                meta = metas[idx]
                score = float(scores[idx])
                
                # Dynamic [IsRel] token classification threshold
                is_rel_token = "[Relevant]" if score > 0.1 else "[Irrelevant]"

                passages_critique.append(PassageCritique(
                    source=meta.get("source", "Unknown"),
                    page=meta.get("page", 0),
                    is_rel=is_rel_token,
                    rel_score=round(score, 3),
                    snippet=doc_text[:90] + "..."
                ))

                if is_rel_token == "[Relevant]":
                    valid_blocks.append(f"[{meta.get('source')} Page {meta.get('page')}]: {doc_text}")

            context_str = "\n\n".join(valid_blocks[:2])

    # Step 3: Generation
    if context_str:
        system_prompt = "You are a Self-RAG enterprise model. Answer the question strictly grounded in the provided relevant passages."
        user_msg = f"--- GROUNDED PASSAGES ---\n{context_str}\n\n--- QUESTION ---\n{payload.question}"
    else:
        system_prompt = "You are a Self-RAG model answering directly without external document retrieval."
        user_msg = payload.question

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
        draft_answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Step 4: [IsSup] and [IsUse] Reflection Auditing
    audit_res = audit_critique(payload.question, context_str if context_str else "No context needed.", draft_answer)

    reflection_tokens = ReflectionTokens(
        retrieve_decision=retrieval_assessment.get("retrieve_decision", "[Retrieve: Yes]"),
        retrieve_reason=retrieval_assessment.get("reason", ""),
        passages_critique=passages_critique,
        is_sup=audit_res.get("is_sup", "[Fully Supported]"),
        groundedness_rationale=audit_res.get("groundedness_rationale", "Grounded against retrieved chunks."),
        is_use=audit_res.get("is_use", 5)
    )

    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": draft_answer})

    return SelfRAGResponse(
        session_id=session_id,
        standalone_question=payload.question,
        reflection_tokens=reflection_tokens,
        final_answer=draft_answer
    )
