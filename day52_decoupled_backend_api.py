import os
import json
import time
import asyncio
import shutil
import chromadb
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(
    title="Day 52 Production Async Backend Core",
    version="2.0.0",
    description="Decoupled backend with native WebSockets, background ingestion workers, and streaming LLM tokens"
)

# Enterprise CORS Configuration for external frontend apps (React/Vite/Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite Dev Server
        "http://localhost:3000",  # Next.js / Create-React-App
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day52_decoupled_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day52"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DOCUMENT_CATALOG: Dict[str, Dict[str, object]] = {}
SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

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
        "service": "Day 52 Decoupled Production Async Backend",
        "version": "2.0.0",
        "websocket_endpoint": "/ws/chat/{session_id}",
        "indexed_documents": len(DOCUMENT_CATALOG),
        "total_chunks": collection.count()
    }

@app.get("/api/v1/documents", response_model=List[CatalogItem])
def get_indexed_documents():
    """Returns catalog of all processed documents for frontend dashboard."""
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
    """Worker task that extracts text, chunks, vectorizes, and stores embeddings."""
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
async def upload_document_async(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Receives file, saves to disk, and triggers non-blocking background ingestion."""
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

    # Offload extraction & vectorization to async background queue
    background_tasks.add_task(process_pdf_background, file_path, file.filename)

    return IngestionStatusResponse(
        task_id=f"task_{int(time.time())}",
        filename=file.filename,
        status="QUEUED",
        message="Document queued for background parsing and vector indexing."
    )

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat_endpoint(websocket: WebSocket, session_id: str):
    """
    Bi-directional WebSocket endpoint handling real-time token streaming,
    retrieval updates, and structured citations.
    """
    await websocket.accept()
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    try:
        while True:
            # Receive client message frame
            data = await websocket.receive_text()
            payload = json.loads(data)
            question = payload.get("question", "").strip()
            target_doc = payload.get("target_doc", "ALL")

            if not question:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Question cannot be empty."
                }))
                continue

            # Stage 1: Telemetry Event - Retrieval Triggered
            await websocket.send_text(json.dumps({
                "type": "event",
                "stage": "RETRIEVAL_START",
                "message": "Querying dense vector index..."
            }))

            candidates = []
            metas = []
            if collection.count() > 0:
                q_vec = embed_model.encode(question).tolist()
                where_filter = {"source": target_doc} if target_doc != "ALL" else None
                n_res = min(5, collection.count())

                results = collection.query(query_embeddings=[q_vec], n_results=n_res, where=where_filter)
                if results['documents'] and results['documents'][0]:
                    candidates = results['documents'][0]
                    metas = results['metadatas'][0]

            # Stage 2: Telemetry Event - Re-ranking
            await websocket.send_text(json.dumps({
                "type": "event",
                "stage": "RERANKING",
                "message": f"Cross-Encoder scoring across {len(candidates)} candidate chunks..."
            }))

            top_chunks = []
            citations = []
            if candidates:
                pairs = [[question, doc] for doc in candidates]
                scores = reranker.predict(pairs)
                ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]

                for idx in ranked_indices:
                    m = metas[idx]
                    chunk_text = candidates[idx]
                    top_chunks.append(f"[{m.get('source')} Page {m.get('page')}]: {chunk_text}")
                    citations.append({
                        "source": m.get("source"),
                        "page": m.get("page"),
                        "score": round(float(scores[idx]), 3),
                        "snippet": chunk_text[:140] + "..."
                    })

            # Send Citations Frame to Frontend
            await websocket.send_text(json.dumps({
                "type": "citations",
                "data": citations
            }))

            context_str = "\n\n".join(top_chunks) if top_chunks else "No relevant enterprise context found."

            # Stage 3: LLM Token-by-Token Streaming
            await websocket.send_text(json.dumps({
                "type": "event",
                "stage": "STREAMING_START",
                "message": "Synthesizing verified response..."
            }))

            system_prompt = (
                "You are an enterprise AI powered by a decoupled async RAG engine. "
                "Synthesize a clear, verified answer based strictly on the provided context."
            )
            user_msg = f"--- CONTEXT EVIDENCE ---\n{context_str}\n\n--- QUESTION ---\n{question}"

            messages = [{"role": "system", "content": system_prompt}]
            for msg in history[-4:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_msg})

            # Stream tokens directly from Groq
            full_response_text = ""
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
                        full_response_text += delta
                        await websocket.send_text(json.dumps({
                            "type": "token",
                            "delta": delta
                        }))
                        await asyncio.sleep(0.005)  # cooperative yield for smooth UI rendering

            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"LLM Streaming Exception: {str(e)}"
                }))

            # Record turn in persistent session memory
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": full_response_text})

            # Finalize Stream Frame
            await websocket.send_text(json.dumps({
                "type": "done",
                "session_id": session_id,
                "total_tokens_approx": len(full_response_text.split())
            }))

    except WebSocketDisconnect:
        pass
