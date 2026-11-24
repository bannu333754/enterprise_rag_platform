import os
import json
import shutil
import asyncio
import chromadb
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(
    title="Day 32 Streaming RAG API",
    version="1.0.0",
    description="Real-time SSE token streaming RAG with progressive citation injection"
)

# Core Models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day32_streaming_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day32"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class StreamRAGRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 32 Streaming RAG API",
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
        "total_collection_chunks": collection.count()
    }

def retrieve_top_chunks(query: str, target_doc: str = "ALL", limit: int = 2):
    if collection.count() == 0:
        return []

    q_vec = embed_model.encode(query).tolist()
    where_filter = {"source": target_doc} if target_doc != "ALL" else None
    n_res = min(4, collection.count())

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=n_res,
        where=where_filter
    )

    if not results['documents'] or not results['documents'][0]:
        return []

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]

    retrieved = []
    for idx in ranked_idx:
        retrieved.append({
            "source": metas[idx].get("source", "Unknown"),
            "page": metas[idx].get("page", 0),
            "score": round(float(scores[idx]), 3),
            "text": candidates[idx]
        })
    return retrieved

async def event_generator(payload: StreamRAGRequest):
    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    # Step 1: Retrieval & Pre-flight Citations
    evidence = retrieve_top_chunks(payload.question, payload.target_doc)
    
    # Send Citations Metadata as first SSE packet
    preflight_meta = {
        "event": "citations_metadata",
        "citations": evidence
    }
    yield f"data: {json.dumps(preflight_meta)}\n\n"
    await asyncio.sleep(0.01)

    # Step 2: Context String Assembly
    if evidence:
        context_blocks = [f"[{e['source']} Page {e['page']}]: {e['text']}" for e in evidence]
        context_str = "\n\n".join(context_blocks)
        system_prompt = (
            "You are an enterprise AI assistant with streaming capabilities. "
            "Answer the query factually based on the provided context."
        )
        user_content = f"--- RETRIEVED CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{payload.question}"
    else:
        system_prompt = "You are an enterprise AI assistant answering conversational queries directly."
        user_content = payload.question

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_content})

    # Step 3: Stream Tokens
    try:
        response_stream = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.2,
            stream=True
        )

        accumulated_answer = []
        for chunk in response_stream:
            delta = chunk.choices[0].delta.content
            if delta:
                accumulated_answer.append(delta)
                token_event = {
                    "event": "token",
                    "delta": delta
                }
                yield f"data: {json.dumps(token_event)}\n\n"
                await asyncio.sleep(0.005)

        full_answer = "".join(accumulated_answer)
        history.append({"role": "user", "content": payload.question})
        history.append({"role": "assistant", "content": full_answer})

        # Step 4: Completion event
        completion_event = {
            "event": "done",
            "full_length": len(full_answer)
        }
        yield f"data: {json.dumps(completion_event)}\n\n"

    except Exception as e:
        error_event = {
            "event": "error",
            "message": str(e)
        }
        yield f"data: {json.dumps(error_event)}\n\n"

@app.post("/chat/stream")
async def stream_chat(payload: StreamRAGRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    return StreamingResponse(
        event_generator(payload),
        media_type="text/event-stream"
    )
