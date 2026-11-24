import os
import json
import shutil
import networkx as nx
import chromadb
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from groq import Groq

app = FastAPI(
    title="Day 36 Graph-Augmented Entity-Disambiguated RAG API",
    version="1.0.0",
    description="Knowledge graph triplet extraction, entity disambiguation, and subgraph path traversal"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day36_vector_store")

groq_client = Groq(api_key="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")

UPLOAD_DIR = "uploaded_docs_day36"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-Memory Directed Knowledge Graph
KG = nx.DiGraph()
SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}

class GraphRAGRequest(BaseModel):
    session_id: str
    question: str
    target_doc: str = "ALL"
    max_hops: int = 2

class GraphPath(BaseModel):
    source_entity: str
    relation: str
    target_entity: str
    document_source: str

class VectorChunkHit(BaseModel):
    source: str
    page: int
    score: float
    text: str

class GraphRAGResponse(BaseModel):
    session_id: str
    standalone_question: str
    extracted_entities: List[str]
    graph_subgraph_triplets: List[GraphPath]
    vector_evidence: List[VectorChunkHit]
    final_answer: str

@app.get("/")
def home():
    return {
        "status": "Online",
        "service": "Day 36 Graph-Augmented RAG",
        "total_graph_nodes": KG.number_of_nodes(),
        "total_graph_edges": KG.number_of_edges(),
        "total_vector_chunks": collection.count(),
        "active_sessions": len(SESSION_HISTORIES)
    }

def extract_triplets_from_chunk(text: str, source: str) -> List[Tuple[str, str, str]]:
    """Extracts Subject-Predicate-Object triplets for knowledge graph insertion."""
    prompt = f"""You are a Knowledge Graph Information Extraction system.
Extract key factual Subject-Predicate-Object relationships from the text below.
Focus on entities (people, technologies, organizations, systems, dates) and their direct connections.

Text:
{text[:1000]}

Respond ONLY with a JSON array of arrays, for example:
[["EntityA", "DEVELOPED_BY", "EntityB"], ["SystemX", "USES", "ComponentY"]]
JSON:"""
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
        triplets = json.loads(raw.strip())
        valid_triplets = []
        for t in triplets:
            if isinstance(t, list) and len(t) == 3:
                valid_triplets.append((str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()))
        return valid_triplets
    except Exception:
        return []

def extract_entities_from_query(query: str) -> List[str]:
    """Identifies core entities in user query for subgraph anchoring."""
    prompt = f"""Extract 1 to 3 core named entities, key concepts, or acronyms from the question.
Question: "{query}"

Respond ONLY as a valid JSON list of strings, e.g. ["Entity1", "Entity2"]:"""
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
        entities = json.loads(raw.strip())
        return [str(e).strip() for e in entities if isinstance(e, str)]
    except Exception:
        return [w for w in query.split() if len(w) > 4][:2]

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    reader = PdfReader(file_path)
    total_added = 0
    triplets_added = 0

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

            # Extract triplets and update Knowledge Graph
            triplets = extract_triplets_from_chunk(chunk, file.filename)
            for sub, pred, obj in triplets:
                KG.add_node(sub, label="Entity")
                KG.add_node(obj, label="Entity")
                KG.add_edge(sub, obj, relation=pred, source=file.filename)
                triplets_added += 1

    return {
        "status": "Success",
        "filename": file.filename,
        "vector_chunks": total_added,
        "kg_triplets_indexed": triplets_added,
        "current_total_nodes": KG.number_of_nodes(),
        "current_total_edges": KG.number_of_edges()
    }

def traverse_subgraph(entities: List[str], max_hops: int = 2) -> List[GraphPath]:
    """Finds neighboring entity relationships in KG around identified query nodes."""
    paths: List[GraphPath] = []
    seen = set()

    for query_node in entities:
        # Entity matching (case-insensitive node search)
        matched_nodes = [n for n in KG.nodes if query_node.lower() in str(n).lower()]
        for start_node in matched_nodes:
            # 1-Hop Neighbors
            for neighbor in KG.neighbors(start_node):
                edge_data = KG.get_edge_data(start_node, neighbor, default={})
                rel = edge_data.get("relation", "CONNECTED_TO")
                src = edge_data.get("source", "Doc")
                pair_key = (start_node, rel, neighbor)
                if pair_key not in seen:
                    seen.add(pair_key)
                    paths.append(GraphPath(
                        source_entity=str(start_node),
                        relation=rel,
                        target_entity=str(neighbor),
                        document_source=src
                    ))

            # 2-Hop Neighbors
            if max_hops >= 2:
                for neighbor in list(KG.neighbors(start_node)):
                    for second_neighbor in KG.neighbors(neighbor):
                        edge_data = KG.get_edge_data(neighbor, second_neighbor, default={})
                        rel = edge_data.get("relation", "CONNECTED_TO")
                        src = edge_data.get("source", "Doc")
                        pair_key = (neighbor, rel, second_neighbor)
                        if pair_key not in seen and len(paths) < 15:
                            seen.add(pair_key)
                            paths.append(GraphPath(
                                source_entity=str(neighbor),
                                relation=rel,
                                target_entity=str(second_neighbor),
                                document_source=src
                            ))
    return paths

@app.post("/chat", response_model=GraphRAGResponse)
def graph_rag_chat(payload: GraphRAGRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if collection.count() == 0 and KG.number_of_edges() == 0:
        raise HTTPException(status_code=400, detail="No knowledge base initialized. Upload a PDF first.")

    session_id = payload.session_id
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []
    history = SESSION_HISTORIES[session_id]

    # Step 1: Query Entity Extraction & Graph Subgraph Traversal
    detected_entities = extract_entities_from_query(payload.question)
    subgraph_triplets = traverse_subgraph(detected_entities, max_hops=payload.max_hops)

    # Step 2: Dense Vector Retrieval + Cross-Encoder Re-ranking
    q_vec = embed_model.encode(payload.question).tolist()
    where_filter = {"source": payload.target_doc} if payload.target_doc != "ALL" else None
    n_res = min(4, collection.count()) if collection.count() > 0 else 0

    vector_evidence: List[VectorChunkHit] = []
    raw_candidates = []
    metas = []

    if n_res > 0:
        results = collection.query(
            query_embeddings=[q_vec],
            n_results=n_res,
            where=where_filter
        )
        if results['documents'] and results['documents'][0]:
            raw_candidates = results['documents'][0]
            metas = results['metadatas'][0]

            pairs = [[payload.question, doc] for doc in raw_candidates]
            scores = reranker.predict(pairs)
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]

            for idx in ranked_indices:
                m = metas[idx]
                vector_evidence.append(VectorChunkHit(
                    source=m.get("source", "Unknown"),
                    page=m.get("page", 0),
                    score=round(float(scores[idx]), 3),
                    text=raw_candidates[idx]
                ))

    # Step 3: Hybrid Fusion (Graph Subgraph Strings + Text Passages)
    graph_context_lines = [
        f"({p.source_entity}) -[:{p.relation}]-> ({p.target_entity})"
        for p in subgraph_triplets[:10]
    ]
    graph_context_str = "\n".join(graph_context_lines) if graph_context_lines else "No direct graph paths detected."

    vector_context_lines = [
        f"[{v.source} Page {v.page}]:\n{v.text}" for v in vector_evidence
    ]
    vector_context_str = "\n\n".join(vector_context_lines) if vector_context_lines else "No dense vector chunks matched."

    system_prompt = (
        "You are an enterprise knowledge graph AI assistant. Answer the user prompt by synthesizing "
        "both explicit knowledge graph relational triplets and contextual dense passages."
    )
    user_msg = (
        f"--- KNOWLEDGE GRAPH RELATIONAL TRIPLET PATHS ---\n{graph_context_str}\n\n"
        f"--- UNSTRUCTURED TEXT CHUNKS ---\n{vector_context_str}\n\n"
        f"--- QUESTION ---\n{payload.question}"
    )

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

    return GraphRAGResponse(
        session_id=session_id,
        standalone_question=payload.question,
        extracted_entities=detected_entities,
        graph_subgraph_triplets=subgraph_triplets,
        vector_evidence=vector_evidence,
        final_answer=final_answer
    )
