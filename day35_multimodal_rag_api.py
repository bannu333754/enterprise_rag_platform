import os
import shutil
import chromadb
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(
    title="Day 35 Multi-Modal Vision-Text RAG API",
    version="1.0.0",
    description="Dual-stream text and visual-layout grounding with cross-modal fusion"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day35_multimodal_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day35"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class MultiModalRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"
    include_visual_nodes: bool = True

class MultiModalHit(BaseModel):
    source: str
    page: int
    modality: str  # "TEXT" or "VISUAL_TABLE_DIAGRAM"
    score: float
    content: str
    visual_descriptor: Optional[str] = None

class MultiModalResponse(BaseModel):
    session_id: str
    standalone_question: str
    retrieved_items: List[MultiModalHit]
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 35 Multi-Modal Vision-Text RAG",
        "total_modal_chunks": collection.count(),
        "active_sessions": len(SESSION_HISTORIES)
    }

def synthesize_visual_descriptor(page_text: str, page_num: int) -> str:
    """Simulates multi-modal vision synthesis: extracts inferred diagrams, metrics, and tables from layout."""
    prompt = f"""You are a multi-modal document vision interpreter. 
Analyze the following text extracted from Page {page_num} of a technical document. 
Identify and create an explicit visual description of any tables, charts, architecture blocks, or numerical comparisons present.

Document Content:
{page_text[:1200]}

Respond with a concise, structured Visual Layout Descriptor (Diagrams, Tables, Metrics):"""
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return f"Visual structure on Page {page_num}: Structured table and technical components."

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

        # 1. Text Modality Stream
        for c_idx, chunk in enumerate(chunks):
            doc_id = f"{file.filename}_txt_p{page_idx + 1}_c{c_idx}"
            vector = embed_model.encode(chunk).tolist()

            collection.add(
                documents=[chunk],
                embeddings=[vector],
                metadatas=[{
                    "source": file.filename,
                    "page": page_idx + 1,
                    "modality": "TEXT",
                    "descriptor": "Standard Text Passage"
                }],
                ids=[doc_id]
            )
            total_added += 1

        # 2. Visual / Table Descriptor Stream
        visual_summary = synthesize_visual_descriptor(text, page_idx + 1)
        v_doc_id = f"{file.filename}_vis_p{page_idx + 1}"
        v_vector = embed_model.encode(visual_summary).tolist()

        collection.add(
            documents=[visual_summary],
            embeddings=[v_vector],
            metadatas=[{
                "source": file.filename,
                "page": page_idx + 1,
                "modality": "VISUAL_TABLE_DIAGRAM",
                "descriptor": visual_summary[:120]
            }],
            ids=[v_doc_id]
        )
        total_added += 1

    return {
        "status": "Success",
        "filename": file.filename,
        "dual_stream_chunks_indexed": total_added,
        "total_chunks": collection.count()
    }

@app.post("/chat", response_model=MultiModalResponse)
def multimodal_chat(payload: MultiModalRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    q_vec = embed_model.encode(payload.question).tolist()
    where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None
    n_res = min(6, collection.count())

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=n_res,
        where=where_filter
    )

    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No matching text or visual passages found.")

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    # Re-rank text and visual modalities together
    pairs = [[payload.question, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    retrieved_hits: List[MultiModalHit] = []
    context_blocks = []

    for idx in ranked_indices:
        m = metas[idx]
        modality = m.get("modality", "TEXT")
        if not payload.include_visual_nodes and modality == "VISUAL_TABLE_DIAGRAM":
            continue

        sc = round(float(scores[idx]), 3)
        doc_text = candidates[idx]

        retrieved_hits.append(MultiModalHit(
            source=m.get("source", "Unknown"),
            page=m.get("page", 0),
            modality=modality,
            score=sc,
            content=doc_text,
            visual_descriptor=m.get("descriptor")
        ))
        
        prefix = "🖼️ [VISUAL/TABLE DESCRIPTOR]" if modality == "VISUAL_TABLE_DIAGRAM" else "📄 [TEXT PASSAGE]"
        context_blocks.append(f"{prefix} ({m.get('source')} Page {m.get('page')}):\n{doc_text}")

    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an enterprise multi-modal AI assistant. Answer user queries comprehensively, "
        "synthesizing data from both textual descriptions and visual/table structural representations."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"--- RETRIEVED DUAL-STREAM CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{payload.question}"
    })

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

    return MultiModalResponse(
        session_id=session_id,
        standalone_question=payload.question,
        retrieved_items=retrieved_hits,
        final_answer=final_answer
    )
