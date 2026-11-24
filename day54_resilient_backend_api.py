import os
import json
import time
import asyncio
import shutil
import chromadb
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(
    title="Day 54 Resilient Enterprise RAG Core",
    version="2.1.0",
    description="Backpressure throttling, sequence-numbered frames, SSE fallback, and state recovery"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day54_resilient_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day54"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DOCUMENT_CATALOG: Dict[str, Dict[str, object]] = {}
SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

# Session Stream Buffers for Reconnection Recovery
STREAM_BUFFERS: Dict[str, List[Dict[str, object]]] = {}

class IngestionStatusResponse(BaseModel):
    task_id: str
    filename: str
    status: str
    message: str

class CatalogItem(BaseModel):
    filename: str
    status: str
    total_chunks: int
    uploaded_at: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 54 Resilient Streaming Engine",
        "supported_transports": ["WebSocket", "Server-Sent Events (SSE)"],
        "indexed_documents": len(DOCUMENT_CATALOG),
        "total_chunks": collection.count()
    }

@app.get("/api/v1/documents", response_model=List[CatalogItem])
def get_documents():
    catalog = []
    for fname, data in DOCUMENT_CATALOG.items():
        catalog.append(CatalogItem(
            filename=fname,
            status=str(data.get("status", "READY")),
            total_chunks=int(data.get("chunks", 0)),
            uploaded_at=str(data.get("timestamp", ""))
        ))
    return catalog

def process_pdf_background(file_path: str, filename: str):
    try:
        DOCUMENT_CATALOG[filename]["status"] = "PROCESSING"
        reader = PdfReader(file_path)
        total_chunks = 0

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue

            words = text.split()
            chunk_size = 75
            chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

            for c_idx, chunk in enumerate(chunks):
                doc_id = f"{filename}_p{page_idx + 1}_c{c_idx}"
                vector = embed_model.encode(chunk).tolist()

                collection.add(
                    documents=[chunk],
                    embeddings=[vector],
                    metadatas=[{"source": filename, "page": page_idx + 1}],
                    ids=[doc_id]
                )
                total_chunks += 1

        DOCUMENT_CATALOG[filename]["status"] = "COMPLETED"
        DOCUMENT_CATALOG[filename]["chunks"] = total_chunks
    except Exception as e:
        DOCUMENT_CATALOG[filename]["status"] = f"FAILED: {str(e)}"

@app.post("/api/v1/upload", response_model=IngestionStatusResponse, status_code=202)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    DOCUMENT_CATALOG[file.filename] = {
        "status": "QUEUED",
        "chunks": 0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    background_tasks.add_task(process_pdf_background, file_path, file.filename)
    return IngestionStatusResponse(
        task_id=f"task_{int(time.time())}",
        filename=file.filename,
        status="QUEUED",
        message="Queued for resilient background ingestion."
    )

def perform_retrieval_and_rerank(question: str, target_doc: str = "ALL"):
    if collection.count() == 0:
        return [], []

    q_vec = embed_model.encode(question).tolist()
    where_filter = {"source": target_doc} if target_doc != "ALL" else None
    n_res = min(6, collection.count())

    results = collection.query(query_embeddings=[q_vec], n_results=n_res, where=where_filter)
    if not results['documents'] or not results['documents'][0]:
        return [], []

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    pairs = [[question, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    top_chunks = []
    citations = []
    for idx in ranked_indices:
        m = metas[idx]
        top_chunks.append(f"[{m.get('source')} Page {m.get('page')}]: {candidates[idx]}")
        citations.append({
            "source": m.get("source"),
            "page": m.get("page"),
            "score": round(float(scores[idx]), 3),
            "snippet": candidates[idx][:140] + "..."
        })

    return top_chunks, citations

# ----------------- Transport 1: WebSocket with Sequence Numbering -----------------
@app.websocket("/ws/chat/{session_id}")
async def websocket_resilient_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    if session_id not in STREAM_BUFFERS:
        STREAM_BUFFERS[session_id] = []

    history = SESSION_HISTORIES[session_id]
    seq_counter = 0

    async def emit_frame(payload_type: str, data: dict):
        nonlocal seq_counter
        seq_counter += 1
        frame = {
            "seq_id": seq_counter,
            "session_id": session_id,
            "timestamp": time.time(),
            "type": payload_type,
            **data
        }
        STREAM_BUFFERS[session_id].append(frame)
        await websocket.send_text(json.dumps(frame))

    try:
        while True:
            raw = await websocket.receive_text()
            req = json.loads(raw)

            # Check if this is a buffer replay request after reconnect
            if req.get("type") == "reconnect_replay":
                last_seen_seq = req.get("last_seen_seq", 0)
                replays = [f for f in STREAM_BUFFERS[session_id] if f["seq_id"] > last_seen_seq]
                for rf in replays:
                    await websocket.send_text(json.dumps(rf))
                continue

            question = req.get("question", "").strip()
            target_doc = req.get("target_doc", "ALL")

            if not question:
                await emit_frame("error", {"message": "Question cannot be empty."})
                continue

            # Stage 1: Retrieval
            await emit_frame("event", {"stage": "RETRIEVAL", "message": "Querying ChromaDB vector space..."})
            top_chunks, citations = perform_retrieval_and_rerank(question, target_doc)

            # Stage 2: Citations Dispatch
            await emit_frame("citations", {"citations": citations})

            # Stage 3: LLM Stream Synthesis with Bounded Backpressure
            await emit_frame("event", {"stage": "STREAMING", "message": "Streaming tokens with backpressure..."})

            context_str = "\n\n".join(top_chunks) if top_chunks else "No relevant enterprise context."
            system_prompt = "You are an enterprise AI. Answer factually and concisely using provided context."
            user_msg = f"--- CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{question}"

            messages = [{"role": "system", "content": system_prompt}]
            for msg in history[-4:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_msg})

            full_text = ""
            try:
                stream = groq_client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=messages,
                    temperature=0.2,
                    stream=True
                )

                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        full_text += delta
                        await emit_frame("token", {"delta": delta})
                        # Backpressure micro-yield: avoids socket buffer flooding
                        await asyncio.sleep(0.008)

            except Exception as e:
                await emit_frame("error", {"message": f"LLM error: {str(e)}"})

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": full_text})

            await emit_frame("done", {
                "total_tokens": len(full_text.split()),
                "status": "SUCCESS"
            })

    except WebSocketDisconnect:
        pass

# ----------------- Transport 2: HTTP Server-Sent Events (SSE Fallback) -----------------
@app.get("/api/v1/stream/{session_id}")
async def sse_fallback_stream(
    session_id: str,
    question: str = Query(...),
    target_doc: str = Query("ALL")
):
    """Fallback streaming route for environments where WebSockets are blocked."""
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    async def event_generator():
        seq = 0
        def make_sse(ptype: str, data: dict):
            nonlocal seq
            seq += 1
            payload = {"seq_id": seq, "type": ptype, **data}
            return f"data: {json.dumps(payload)}\n\n"

        yield make_sse("event", {"stage": "RETRIEVAL", "message": "SSE: Querying vector store..."})
        top_chunks, citations = perform_retrieval_and_rerank(question, target_doc)
        yield make_sse("citations", {"citations": citations})

        yield make_sse("event", {"stage": "STREAMING", "message": "SSE: Synthesizing tokens..."})

        context_str = "\n\n".join(top_chunks) if top_chunks else "No relevant enterprise context."
        system_prompt = "You are an enterprise AI. Answer factually based on context."
        user_msg = f"--- CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{question}"

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_msg})

        full_text = ""
        try:
            stream = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                temperature=0.2,
                stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_text += delta
                    yield make_sse("token", {"delta": delta})
                    await asyncio.sleep(0.008)
        except Exception as e:
            yield make_sse("error", {"message": str(e)})

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": full_text})
        yield make_sse("done", {"status": "SUCCESS"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
