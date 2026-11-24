import time
import uuid
import hmac
import hashlib
import asyncio
import logging
from enum import Enum
from typing import List, Dict, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from sentence_transformers import SentenceTransformer
import chromadb
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [INGEST] %(message)s")
logger = logging.getLogger("AsyncIngestion")

app = FastAPI(
    title="Day 64 Enterprise Asynchronous Document Ingestion Engine",
    version="8.0.0",
    description="Decoupled background ingestion worker, batch embedding pipeline, and HMAC signed webhooks"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Job State Machine & In-Memory Task Registry -----------------
class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"

class IngestionJob(BaseModel):
    job_id: str
    filename: str
    status: JobStatus
    progress_percentage: int
    total_chunks: int
    created_at: float
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    webhook_url: Optional[str] = None

# In-memory broker/state registry (simulates Redis Key-Value store)
JOB_REGISTRY: Dict[str, IngestionJob] = {}
WEBHOOK_SECRET = "secret_enterprise_ingest_key_2026"

# ----------------- Embedding Model & Vector Storage -----------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day64_async_rag")

# ----------------- Chunking Utility -----------------
def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# ----------------- Worker Task Execution -----------------
def execute_background_ingestion(job_id: str, document_text: str, webhook_url: Optional[str]):
    job = JOB_REGISTRY.get(job_id)
    if not job:
        return

    try:
        job.status = JobStatus.PROCESSING
        job.progress_percentage = 20
        logger.info(f"Job {job_id} -> PROCESSING: Chunking document '{job.filename}'...")
        time.sleep(0.5)

        # 1. Chunking stage
        chunks = chunk_text(document_text)
        job.total_chunks = len(chunks)
        job.progress_percentage = 50
        logger.info(f"Job {job_id} -> Generated {len(chunks)} chunks.")

        # 2. Batch Embedding Generation
        time.sleep(0.5)
        embeddings = embed_model.encode(chunks, batch_size=32).tolist()
        job.progress_percentage = 80

        # 3. Vector Database Commit
        doc_ids = [f"{job_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"job_id": job_id, "filename": job.filename, "chunk_index": i} for i in range(len(chunks))]
        collection.add(documents=chunks, embeddings=embeddings, ids=doc_ids, metadatas=metadatas)

        job.status = JobStatus.INDEXED
        job.progress_percentage = 100
        job.completed_at = time.time()
        logger.info(f"Job {job_id} -> Successfully INDEXED into ChromaDB.")

        # 4. HMAC Signed Webhook Delivery
        if webhook_url:
            dispatch_webhook(job, webhook_url)

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error_message = str(e)
        logger.error(f"Job {job_id} execution failed: {e}")

def dispatch_webhook(job: IngestionJob, webhook_url: str):
    payload = {
        "event": "document.indexed",
        "job_id": job.job_id,
        "filename": job.filename,
        "total_chunks": job.total_chunks,
        "timestamp": time.time()
    }
    encoded_payload = str(payload).encode("utf-8")
    signature = hmac.new(WEBHOOK_SECRET.encode("utf-8"), encoded_payload, hashlib.sha256).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-Hub-Signature-256": f"sha256={signature}"
    }

    try:
        requests.post(webhook_url, json=payload, headers=headers, timeout=2.0)
        logger.info(f"Webhook dispatched to {webhook_url} with signature {signature[:12]}...")
    except Exception as e:
        logger.warning(f"Failed to deliver webhook to {webhook_url}: {e}")

# ----------------- API Endpoints -----------------
class IngestionRequest(BaseModel):
    filename: str = Field(..., example="annual_cloud_report_2026.txt")
    content: str = Field(..., min_length=20, example="Enterprise architecture migration guide...")
    webhook_url: Optional[str] = Field(default=None, example="http://127.0.0.1:8000/api/v1/mock-webhook-receiver")

class IngestionAcceptedResponse(BaseModel):
    job_id: str
    status: str
    message: str
    poll_endpoint: str

@app.post("/api/v1/ingest/async", status_code=status.HTTP_202_ACCEPTED, response_model=IngestionAcceptedResponse)
async def submit_ingestion_job(payload: IngestionRequest, background_tasks: BackgroundTasks):
    job_id = f"job_{uuid.uuid4().hex[:8]}"

    job = IngestionJob(
        job_id=job_id,
        filename=payload.filename,
        status=JobStatus.PENDING,
        progress_percentage=0,
        total_chunks=0,
        created_at=time.time(),
        webhook_url=payload.webhook_url
    )
    JOB_REGISTRY[job_id] = job

    # Dispatch to background task runner (simulates Celery queue task)
    background_tasks.add_task(execute_background_ingestion, job_id, payload.content, payload.webhook_url)

    logger.info(f"Accepted ingestion job '{job_id}' for file '{payload.filename}'")
    return IngestionAcceptedResponse(
        job_id=job_id,
        status="PENDING",
        message="Document queued for background chunking and vector indexing.",
        poll_endpoint=f"/api/v1/ingest/status/{job_id}"
    )

@app.get("/api/v1/ingest/status/{job_id}", response_model=IngestionJob)
async def get_job_status(job_id: str):
    job = JOB_REGISTRY.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return job

# Mock webhook receiver endpoint for local test verification
WEBHOOK_RECEIVER_LOG: List[Dict] = []

@app.post("/api/v1/mock-webhook-receiver")
async def mock_webhook_receiver(data: Dict):
    WEBHOOK_RECEIVER_LOG.append(data)
    logger.info(f"Mock receiver received webhook callback for job: {data.get('job_id')}")
    return {"received": True}

@app.get("/api/v1/mock-webhook-receiver/logs")
def get_webhook_logs():
    return {"count": len(WEBHOOK_RECEIVER_LOG), "logs": WEBHOOK_RECEIVER_LOG}
