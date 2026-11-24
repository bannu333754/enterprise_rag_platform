import os
import shutil
import numpy as np
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from groq import Groq

app = FastAPI(title="Day 27 Late-Interaction ColBERT-style RAG API", version="1.0.0")

# Token-level representation embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day27"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-Memory Token Matrix Store: doc_id -> { "source": str, "page": int, "text": str, "token_matrix": np.ndarray, "tokens": List[str] }
TOKEN_STORE: Dict[str, Dict] = {}
SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class LateInteractionRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class TokenMatchExplanation(BaseModel):
    query_token: str
    matched_doc_token: str
    similarity_score: float

class LateRetrievedItem(BaseModel):
    doc_id: str
    source: str
    page: int
    maxsim_score: float
    content: str
    token_alignments: List[TokenMatchExplanation]

class LateInteractionResponse(BaseModel):
    session_id: str
    standalone_question: str
    retrieved_items: List[LateRetrievedItem]
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 27 Late-Interaction Multi-Vector RAG",
        "total_documents_indexed": len(TOKEN_STORE),
        "active_sessions": len(SESSION_HISTORIES)
    }

def get_token_embeddings(text: str):
    """Generates normalized per-token embeddings for late-interaction matrix alignment."""
    tokens = text.split()
    if not tokens:
        tokens = ["empty"]
    embeddings = embed_model.encode(tokens, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalized_matrix = embeddings / norms
    return tokens, normalized_matrix

def compute_maxsim(query_tokens: List[str], query_mat: np.ndarray, doc_tokens: List[str], doc_mat: np.ndarray):
    """Computes ColBERT-style MaxSim: sum over query tokens of max dot-product with doc tokens."""
    # Dot product matrix: shape (len(query_tokens), len(doc_tokens))
    sim_matrix = np.dot(query_mat, doc_mat.T)
    max_scores = np.max(sim_matrix, axis=1)
    best_doc_indices = np.argmax(sim_matrix, axis=1)
    
    total_score = float(np.sum(max_scores))
    
    explanations = []
    for q_idx, q_tok in enumerate(query_tokens[:5]):  # Top query alignments for observability
        d_idx = best_doc_indices[q_idx]
        score = float(max_scores[q_idx])
        explanations.append(TokenMatchExplanation(
            query_token=q_tok,
            matched_doc_token=doc_tokens[d_idx] if d_idx < len(doc_tokens) else "N/A",
            similarity_score=round(score, 3)
        ))
    return total_score, explanations

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
        chunk_size = 65
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        for c_idx, chunk in enumerate(chunks):
            doc_id = f"{file.filename}_p{page_idx + 1}_c{c_idx}"
            tokens, token_mat = get_token_embeddings(chunk)

            TOKEN_STORE[doc_id] = {
                "source": file.filename,
                "page": page_idx + 1,
                "text": chunk,
                "tokens": tokens,
                "token_matrix": token_mat
            }
            total_added += 1

    return {
        "status": "Success",
        "filename": file.filename,
        "token_matrices_indexed": total_added,
        "total_store_documents": len(TOKEN_STORE)
    }

@app.post("/chat", response_model=LateInteractionResponse)
def late_interaction_chat(payload: LateInteractionRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(TOKEN_STORE) == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    # Compute Query Token Embeddings
    q_tokens, q_matrix = get_token_embeddings(payload.question)

    scored_candidates = []
    for doc_id, doc_data in TOKEN_STORE.items():
        if payload.target_doc != "ALL" and doc_data["source"] != payload.target_doc:
            continue

        score, explanations = compute_maxsim(
            q_tokens, q_matrix,
            doc_data["tokens"], doc_data["token_matrix"]
        )
        scored_candidates.append({
            "doc_id": doc_id,
            "source": doc_data["source"],
            "page": doc_data["page"],
            "score": score,
            "text": doc_data["text"],
            "explanations": explanations
        })

    if not scored_candidates:
        raise HTTPException(status_code=404, detail="No matching document passages found.")

    # Rank by MaxSim cumulative score
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = scored_candidates[:2]

    retrieved_items: List[LateRetrievedItem] = []
    context_blocks = []

    for item in top_candidates:
        retrieved_items.append(LateRetrievedItem(
            doc_id=item["doc_id"],
            source=item["source"],
            page=item["page"],
            maxsim_score=round(item["score"], 3),
            content=item["text"],
            token_alignments=item["explanations"]
        ))
        context_blocks.append(f"[{item['source']} Page {item['page']}]:\n{item['text']}")

    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an enterprise research assistant using token-aligned Late-Interaction RAG. "
        "Answer the question directly and factually using the retrieved grounded passages."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"--- CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{payload.question}"
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

    return LateInteractionResponse(
        session_id=session_id,
        standalone_question=payload.question,
        retrieved_items=retrieved_items,
        final_answer=final_answer
    )
