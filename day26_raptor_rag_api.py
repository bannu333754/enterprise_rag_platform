import os
import shutil
import chromadb
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(title="Day 26 RAPTOR Recursive Tree RAG API", version="1.0.0")

# Embedding & CrossEncoder Models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day26_raptor_tree_rag")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day26"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class RaptorRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"

class TreeRetrievedItem(BaseModel):
    node_id: str
    layer_type: str  # "LEAF" or "TREE_SUMMARY"
    source: str
    score: float
    content: str

class RaptorResponse(BaseModel):
    session_id: str
    standalone_question: str
    retrieved_nodes: List[TreeRetrievedItem]
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 26 RAPTOR Recursive Tree RAG",
        "total_tree_nodes": collection.count(),
        "active_sessions": len(SESSION_HISTORIES)
    }

def summarize_cluster(chunk_list: List[str]) -> str:
    """Recursively aggregates and abstracts a cluster of leaf chunks into a higher-order tree node."""
    combined_text = "\n---\n".join(chunk_list)
    prompt = f"""You are a high-level enterprise conceptual architect. 
Analyze and synthesize the following connected document segments into a cohesive, high-level abstractive summary capturing macro-themes, cross-sectional conclusions, and system implications:

Segments:
{combined_text}

Executive Thematic Summary:"""
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return " ".join([c[:100] for c in chunk_list]) + "..."

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    reader = PdfReader(file_path)
    leaf_chunks = []
    total_added = 0

    # 1. Extract Leaf Chunks (Level 0)
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        words = text.split()
        chunk_size = 70
        page_chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        for c_idx, chunk in enumerate(page_chunks):
            doc_id = f"{file.filename}_leaf_p{page_idx + 1}_c{c_idx}"
            vector = embed_model.encode(chunk).tolist()

            collection.add(
                documents=[chunk],
                embeddings=[vector],
                metadatas=[{
                    "source": file.filename,
                    "layer": "LEAF",
                    "page": page_idx + 1
                }],
                ids=[doc_id]
            )
            leaf_chunks.append(chunk)
            total_added += 1

    # 2. Build RAPTOR Tree Nodes (Level 1 Hierarchical Cluster Summaries)
    cluster_size = 3
    for cluster_idx in range(0, len(leaf_chunks), cluster_size):
        cluster_subset = leaf_chunks[cluster_idx:cluster_idx + cluster_size]
        if not cluster_subset:
            continue

        summary_node = summarize_cluster(cluster_subset)
        summary_id = f"{file.filename}_tree_lvl1_cl{cluster_idx // cluster_size}"
        summary_vec = embed_model.encode(summary_node).tolist()

        collection.add(
            documents=[summary_node],
            embeddings=[summary_vec],
            metadatas=[{
                "source": file.filename,
                "layer": "TREE_SUMMARY",
                "page": 0
            }],
            ids=[summary_id]
        )
        total_added += 1

    return {
        "status": "Success",
        "filename": file.filename,
        "total_nodes_indexed": total_added,
        "total_collection_nodes": collection.count()
    }

@app.post("/chat", response_model=RaptorResponse)
def raptor_chat(payload: RaptorRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    query_vec = embed_model.encode(payload.question).tolist()
    where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None
    n_res = min(6, collection.count())

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n_res,
        where=where_filter
    )

    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No matching knowledge nodes found.")

    candidates = results['documents'][0]
    metas = results['metadatas'][0]
    ids = results['ids'][0]

    # Re-rank combining leaf-level specifics and tree-level thematic summaries
    pairs = [[payload.question, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    retrieved_nodes: List[TreeRetrievedItem] = []
    context_blocks = []

    for idx in ranked_indices:
        meta = metas[idx]
        layer = meta.get("layer", "LEAF")
        doc_text = candidates[idx]
        score = float(scores[idx])

        retrieved_nodes.append(TreeRetrievedItem(
            node_id=ids[idx],
            layer_type=layer,
            source=meta.get("source", "Unknown"),
            score=round(score, 3),
            content=doc_text
        ))
        context_blocks.append(f"[{layer} from {meta.get('source')}]:\n{doc_text}")

    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an enterprise knowledge intelligence assistant. Answer the user prompt comprehensively, "
        "leveraging both granular leaf facts and macro tree summaries provided in the context."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"--- RETRIEVED HIERARCHICAL CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{payload.question}"
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

    return RaptorResponse(
        session_id=session_id,
        standalone_question=payload.question,
        retrieved_nodes=retrieved_nodes,
        final_answer=final_answer
    )
