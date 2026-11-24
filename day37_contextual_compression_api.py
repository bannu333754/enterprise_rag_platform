import os
import re
import shutil
import chromadb
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(
    title="Day 37 Contextual Compression & Token Pruning RAG API",
    version="1.0.0",
    description="Sentence-level Cross-Encoder compression, noise pruning, and dense context repacking"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day37_compressed_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day37"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class CompressionRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"
    relevance_threshold: float = 0.1  # Pruning cutoff for sentence score
    max_repacked_sentences: int = 6

class CompressedSentence(BaseModel):
    source: str
    page: int
    score: float
    sentence: str

class CompressionTelemetry(BaseModel):
    raw_character_count: int
    compressed_character_count: int
    pruned_sentences_count: int
    kept_sentences_count: int
    compression_ratio_pct: float

class CompressionResponse(BaseModel):
    session_id: str
    standalone_question: str
    telemetry: CompressionTelemetry
    extracted_sentences: List[CompressedSentence]
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 37 Contextual Compression RAG",
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
        chunk_size = 90
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

def split_into_sentences(text: str) -> List[str]:
    """Splits raw text into individual clean sentences."""
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = [s.strip() for s in raw_sentences if len(s.strip()) > 20]
    return cleaned

@app.post("/chat", response_model=CompressionResponse)
def compressed_rag_chat(payload: CompressionRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    # Step 1: Initial Retrieval of Candidate Chunks
    q_vec = embed_model.encode(payload.question).tolist()
    where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None
    n_res = min(5, collection.count())

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=n_res,
        where=where_filter
    )

    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No candidate passages found.")

    candidates = results['documents'][0]
    metas = results['metadatas'][0]

    raw_text_total = "".join(candidates)
    raw_char_count = len(raw_text_total)

    # Step 2: Sentence-Level Decomposition
    candidate_sentences = []
    for doc, meta in zip(candidates, metas):
        sents = split_into_sentences(doc)
        for s in sents:
            candidate_sentences.append({
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", 0),
                "sentence": s
            })

    if not candidate_sentences:
        raise HTTPException(status_code=404, detail="Could not extract valid sentences from retrieved passages.")

    # Step 3: Sentence Cross-Encoder Scoring & Pruning
    sentence_pairs = [[payload.question, item["sentence"]] for item in candidate_sentences]
    sentence_scores = reranker.predict(sentence_pairs)

    scored_sentences = []
    for idx, item in enumerate(candidate_sentences):
        score = float(sentence_scores[idx])
        scored_sentences.append({
            "source": item["source"],
            "page": item["page"],
            "score": round(score, 3),
            "sentence": item["sentence"]
        })

    # Sort descending by relevance score
    scored_sentences.sort(key=lambda x: x["score"], reverse=True)

    # Filter by threshold and take top K
    kept = [s for s in scored_sentences if s["score"] >= payload.relevance_threshold][:payload.max_repacked_sentences]
    
    # Fallback to top 2 if threshold was too aggressive
    if not kept and scored_sentences:
        kept = scored_sentences[:2]

    pruned_count = len(scored_sentences) - len(kept)
    compressed_text_total = " ".join([k["sentence"] for k in kept])
    compressed_char_count = len(compressed_text_total)

    compression_pct = round(max(0.0, (1.0 - (compressed_char_count / max(1, raw_char_count))) * 100), 2)

    extracted_items = [
        CompressedSentence(
            source=k["source"],
            page=k["page"],
            score=k["score"],
            sentence=k["sentence"]
        ) for k in kept
    ]

    telemetry = CompressionTelemetry(
        raw_character_count=raw_char_count,
        compressed_character_count=compressed_char_count,
        pruned_sentences_count=pruned_count,
        kept_sentences_count=len(kept),
        compression_ratio_pct=compression_pct
    )

    # Step 4: Dense Context Repacking & Synthesis
    repacked_blocks = [f"[{k['source']} p.{k['page']}]: {k['sentence']}" for k in kept]
    dense_context_str = "\n".join(repacked_blocks)

    system_prompt = (
        "You are an enterprise research assistant provided with high-density, pruned context. "
        "Answer the user prompt directly and factually using only the verified compressed sentences."
    )
    user_msg = f"--- COMPRESSED HIGH-DENSITY CONTEXT ---\n{dense_context_str}\n\n--- QUESTION ---\n{payload.question}"

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

    return CompressionResponse(
        session_id=session_id,
        standalone_question=payload.question,
        telemetry=telemetry,
        extracted_sentences=extracted_items,
        final_answer=final_answer
    )
