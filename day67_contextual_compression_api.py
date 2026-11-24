import time
import logging
import re
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
import chromadb
from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [COMPRESS] %(message)s")
logger = logging.getLogger("ContextualCompression")

app = FastAPI(
    title="Day 67 Enterprise Contextual Compression & Sentence-Window Engine",
    version="10.0.0",
    description="Sentence-window retrieval paired with post-retrieval extractive contextual compression"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="day67_sentence_window_rag")

RAW_TECHNICAL_CORPUS = [
    (
        "Kubernetes clusters utilize an internal DNS service called CoreDNS for service discovery across pods. "
        "CoreDNS runs as a deployment and dynamically updates DNS records when services scale. "
        "If CoreDNS pods crash or encounter OOMKilled events, cross-pod communication fails immediately. "
        "Administrators should tune CoreDNS memory limits to at least 170Mi for clusters handling over 500 pods. "
        "Monitoring CoreDNS request duration metrics in Prometheus helps identify cluster-wide network bottlenecks early."
    ),
    (
        "Redis cluster deployments rely on hash slot partitioning across 16384 total virtual buckets. "
        "Each Redis master node handles a specific subset of hash slots to distribute key writes evenly. "
        "When key migration fails during rebalancing, the cluster status changes to CLUSTER_FAIL. "
        "Manual intervention using the redis-cli cluster check command is required to repair unassigned hash slots. "
        "Setting replica-read-only configuration ensures replicas serve read traffic without risking split-brain writes."
    )
]

def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def seed_sentence_window_index():
    if collection.count() > 0:
        return

    documents = []
    ids = []
    metadatas = []

    for doc_idx, doc_text in enumerate(RAW_TECHNICAL_CORPUS):
        sentences = split_into_sentences(doc_text)
        total_s = len(sentences)

        for s_idx, sentence in enumerate(sentences):
            prev_s = sentences[s_idx - 1] if s_idx > 0 else ""
            next_s = sentences[s_idx + 1] if s_idx < total_s - 1 else ""
            
            window_parts = [p for p in [prev_s, sentence, next_s] if p]
            expanded_window = " ".join(window_parts)

            doc_id = f"doc_{doc_idx}_sent_{s_idx}"
            documents.append(sentence)
            ids.append(doc_id)
            metadatas.append({
                "doc_index": doc_idx,
                "sentence_index": s_idx,
                "expanded_window": expanded_window,
                "focal_sentence": sentence
            })

    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    logger.info(f"Seeded {len(documents)} sentence-window units into ChromaDB.")

seed_sentence_window_index()
groq_client = Groq(api_key="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")

def compress_context(query: str, expanded_windows: List[str], similarity_cutoff: float = 0.40) -> Tuple[str, int, int]:
    raw_text = " ".join(expanded_windows)
    raw_tokens = len(raw_text.split())

    candidate_sentences = []
    seen = set()
    for win in expanded_windows:
        for sent in split_into_sentences(win):
            if sent not in seen:
                candidate_sentences.append(sent)
                seen.add(sent)

    if not candidate_sentences:
        return "", raw_tokens, 0

    q_vec = embed_model.encode(query)
    s_vecs = embed_model.encode(candidate_sentences)
    scores = util.cos_sim(q_vec, s_vecs)[0].tolist()

    pruned_sentences = [
        sent for sent, score in zip(candidate_sentences, scores)
        if score >= similarity_cutoff
    ]

    if not pruned_sentences:
        best_idx = scores.index(max(scores))
        pruned_sentences = [candidate_sentences[best_idx]]

    compressed_text = " ".join(pruned_sentences)
    compressed_tokens = len(compressed_text.split())

    return compressed_text, raw_tokens, compressed_tokens

class CompressionQueryRequest(BaseModel):
    question: str = Field(..., example="What memory limit is recommended for CoreDNS?")
    enable_compression: bool = Field(default=True)
    similarity_cutoff: float = Field(default=0.42, ge=0.2, le=0.8)

class CompressionQueryResponse(BaseModel):
    question: str
    answer: str
    raw_context_tokens: int
    compressed_context_tokens: int
    compression_ratio_percent: float
    final_context_used: str

@app.post("/api/v1/contextual-query", response_model=CompressionQueryResponse)
async def contextual_compression_query(payload: CompressionQueryRequest):
    q_str = payload.question.strip()
    if not q_str:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    q_vec = embed_model.encode(q_str).tolist()
    results = collection.query(query_embeddings=[q_vec], n_results=3)

    retrieved_metas = results["metadatas"][0] if results["metadatas"] else []
    expanded_windows = [m["expanded_window"] for m in retrieved_metas]

    if payload.enable_compression:
        final_context, raw_tokens, comp_tokens = compress_context(
            q_str, expanded_windows, similarity_cutoff=payload.similarity_cutoff
        )
        saved_ratio = round(((raw_tokens - comp_tokens) / max(raw_tokens, 1)) * 100, 2)
    else:
        final_context = " ".join(expanded_windows)
        raw_tokens = len(final_context.split())
        comp_tokens = raw_tokens
        saved_ratio = 0.0

    prompt = (
        f"Context:\n{final_context}\n\n"
        f"Question: {q_str}\n"
        f"Provide a direct, factual answer in one concise sentence:"
    )

    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=120
    )
    answer = resp.choices[0].message.content

    return CompressionQueryResponse(
        question=q_str,
        answer=answer,
        raw_context_tokens=raw_tokens,
        compressed_context_tokens=comp_tokens,
        compression_ratio_percent=saved_ratio,
        final_context_used=final_context
    )
