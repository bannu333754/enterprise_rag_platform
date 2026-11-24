import time
import random
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [PROD_APP] %(message)s")
logger = logging.getLogger("EnterpriseProductionApp")

app = FastAPI(
    title="Day 74-75 Enterprise Resilient Production RAG Suite",
    version="17.0.0",
    description="Unified Enterprise RAG Platform with Chaos Engineering, Circuit Breakers, and Interactive Management UI"
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
collection = chroma_client.get_or_create_collection(name="production_enterprise_kb")
groq_client = Groq(api_key="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")

CHAOS_CONFIG = {
    "latency_injection_ms": 0,
    "error_rate_percent": 0.0,
    "circuit_breaker_tripped": False,
    "failure_count": 0,
    "failure_threshold": 3
}

CORPUS = [
    {"id": "doc_arch", "text": "Enterprise RAG Architecture integrates hybrid search, semantic caching with Redis, contextual compression, and distributed OpenTelemetry tracing across all inference stages."},
    {"id": "doc_security", "text": "Zero-trust API security enforces token-bucket rate limiting at the Envoy Gateway, mTLS encryption on port 8443, and HMAC-signed asynchronous document ingestion."},
    {"id": "doc_resilience", "text": "Chaos engineering framework monitors circuit breakers, automated failover pods, and graceful degradation when vector database latency exceeds 250ms."},
    {"id": "doc_general", "text": "In modern software architecture, an enterprise refers to large-scale distributed business platforms requiring high availability, rigorous zero-trust security, resilient failovers, and governance."}
]

def init_kb():
    if collection.count() > 0:
        return
    docs = [c["text"] for c in CORPUS]
    ids = [c["id"] for c in CORPUS]
    collection.add(documents=docs, ids=ids)
    logger.info("Production knowledge base initialized.")

init_kb()

class QueryRequest(BaseModel):
    query: str = Field(..., example="Explain the enterprise zero-trust security and gateway architecture.")

class ChaosSettings(BaseModel):
    latency_ms: int = Field(default=0, ge=0, le=5000)
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    reset_circuit_breaker: bool = False

@app.post("/api/v1/query")
async def execute_query(req: QueryRequest):
    if CHAOS_CONFIG["circuit_breaker_tripped"]:
        raise HTTPException(status_code=503, detail="[CIRCUIT BREAKER OPEN] Upstream vector services degraded. Traffic halted.")

    if CHAOS_CONFIG["latency_injection_ms"] > 0:
        time.sleep(CHAOS_CONFIG["latency_injection_ms"] / 1000.0)

    if CHAOS_CONFIG["error_rate_percent"] > 0:
        if random.random() < CHAOS_CONFIG["error_rate_percent"]:
            CHAOS_CONFIG["failure_count"] += 1
            if CHAOS_CONFIG["failure_count"] >= CHAOS_CONFIG["failure_threshold"]:
                CHAOS_CONFIG["circuit_breaker_tripped"] = True
                logger.error("Circuit breaker TRIPPED due to consecutive chaos failures.")
            raise HTTPException(status_code=500, detail="[CHAOS INJECTION] Simulated service degradation failure.")

    q_str = req.query.strip()
    if not q_str:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    t0 = time.time()
    q_vec = embed_model.encode(q_str).tolist()
    search_res = collection.query(query_embeddings=[q_vec], n_results=2)
    retrieval_ms = round((time.time() - t0) * 1000, 2)

    docs = search_res["documents"][0] if search_res["documents"] else []
    context = "\n".join(docs)

    prompt = (
        f"Context Information:\n{context}\n\n"
        f"User Question: {q_str}\n\n"
        f"Provide a clear, direct, and factual answer in 1-2 complete sentences:"
    )

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Primary model inference fallback: {e}")
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e2:
            answer = context

    total_latency_ms = round((time.time() - t0) * 1000, 2)

    return {
        "query": q_str,
        "answer": answer,
        "retrieved_docs": search_res["ids"][0] if search_res["ids"] else [],
        "retrieval_latency_ms": retrieval_ms,
        "total_latency_ms": total_latency_ms,
        "circuit_status": "CLOSED (HEALTHY)"
    }

@app.post("/api/v1/chaos/configure")
async def configure_chaos(settings: ChaosSettings):
    CHAOS_CONFIG["latency_injection_ms"] = settings.latency_ms
    CHAOS_CONFIG["error_rate_percent"] = settings.error_rate
    if settings.reset_circuit_breaker:
        CHAOS_CONFIG["circuit_breaker_tripped"] = False
        CHAOS_CONFIG["failure_count"] = 0
    return {"status": "updated", "current_chaos_state": CHAOS_CONFIG}

@app.get("/api/v1/chaos/status")
async def get_chaos_status():
    return CHAOS_CONFIG

@app.get("/", response_class=HTMLResponse)
async def dashboard_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Enterprise RAG Platform - Production Showcase</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Inter', sans-serif; background-color: #0b0f19; color: #e2e8f0; }
        </style>
    </head>
    <body class="p-8">
        <div class="max-w-6xl mx-auto space-y-6">
            <div class="flex justify-between items-center border-b border-gray-800 pb-5">
                <div>
                    <h1 class="text-3xl font-extrabold text-white tracking-tight">Enterprise RAG Control Center</h1>
                    <p class="text-sm text-gray-400 mt-1">Days 61-75 Production Architecture Showcase & Chaos Telemetry</p>
                </div>
                <div class="flex items-center space-x-3">
                    <span id="circuitBadge" class="px-3 py-1 text-xs font-semibold rounded-full bg-emerald-900 text-emerald-300 border border-emerald-700">Circuit: HEALTHY</span>
                    <span class="px-3 py-1 text-xs font-semibold rounded-full bg-blue-900 text-blue-300 border border-blue-700">Cluster: v17.0.0</span>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div class="lg:col-span-2 bg-gray-900 border border-gray-800 rounded-xl p-6 shadow-xl space-y-4">
                    <h2 class="text-xl font-bold text-gray-200">Interactive Query Studio</h2>
                    <div>
                        <label class="block text-xs font-semibold text-gray-400 uppercase mb-2">Enterprise Query</label>
                        <textarea id="queryInput" rows="3" class="w-full bg-gray-950 border border-gray-700 rounded-lg p-3 text-white text-sm focus:border-blue-500 focus:outline-none" placeholder="Ask about enterprise architecture, security, or resiliency..."></textarea>
                    </div>
                    <button onclick="runQuery()" id="btnSubmit" class="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-2.5 px-4 rounded-lg transition duration-200 shadow-md">Execute Resilient Query</button>
                    
                    <div id="outputArea" class="hidden mt-4 p-4 bg-gray-950 border border-gray-800 rounded-lg space-y-3">
                        <div class="flex justify-between text-xs text-gray-400 border-b border-gray-800 pb-2">
                            <span>Status: <b class="text-emerald-400">200 OK</b></span>
                            <span>Latency: <b id="latencyBadge" class="text-yellow-400">0 ms</b></span>
                        </div>
                        <div>
                            <span class="text-xs uppercase font-bold text-gray-400">Synthesized Answer</span>
                            <p id="answerText" class="text-gray-100 text-sm mt-1 leading-relaxed bg-gray-900/60 p-3 rounded-lg border border-gray-800"></p>
                        </div>
                        <div>
                            <span class="text-xs uppercase font-bold text-gray-400">Retrieved Chunks</span>
                            <div id="docsList" class="flex gap-2 mt-1"></div>
                        </div>
                    </div>

                    <div id="errorArea" class="hidden mt-4 p-4 bg-red-950/50 border border-red-700 rounded-lg text-red-300 text-sm"></div>
                </div>

                <div class="bg-gray-900 border border-gray-800 rounded-xl p-6 shadow-xl space-y-5">
                    <div class="border-b border-gray-800 pb-3">
                        <h2 class="text-xl font-bold text-red-400">Chaos Injection Panel</h2>
                        <p class="text-xs text-gray-400 mt-1">Inject latency & fault rates to test resiliency</p>
                    </div>

                    <div>
                        <label class="flex justify-between text-xs text-gray-300 font-semibold mb-1">
                            <span>Injected Latency</span>
                            <span id="lblLatency" class="text-blue-400">0 ms</span>
                        </label>
                        <input type="range" id="sliderLatency" min="0" max="3000" step="250" value="0" class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="updateChaosUI()">
                    </div>

                    <div>
                        <label class="flex justify-between text-xs text-gray-300 font-semibold mb-1">
                            <span>Failure Rate</span>
                            <span id="lblError" class="text-red-400">0 %</span>
                        </label>
                        <input type="range" id="sliderError" min="0" max="100" step="10" value="0" class="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer" oninput="updateChaosUI()">
                    </div>

                    <button onclick="applyChaos()" class="w-full bg-red-700 hover:bg-red-600 text-white font-semibold py-2 rounded-lg text-sm transition">Apply Chaos Parameters</button>
                    <button onclick="resetCircuitBreaker()" class="w-full bg-gray-800 hover:bg-gray-700 text-gray-200 font-semibold py-2 rounded-lg text-sm border border-gray-600 transition">Reset Circuit Breaker</button>

                    <div class="p-3 bg-gray-950 border border-gray-800 rounded-lg text-xs space-y-1 text-gray-400">
                        <p><b>Fail Threshold:</b> 3 consecutive errors</p>
                        <p><b>Trigger Action:</b> Trips Circuit Breaker</p>
                        <p><b>Fallbacks:</b> Graceful 503 circuit-open notice</p>
                    </div>
                </div>
            </div>
        </div>

        <script>
            function updateChaosUI() {
                document.getElementById('lblLatency').innerText = document.getElementById('sliderLatency').value + ' ms';
                document.getElementById('lblError').innerText = document.getElementById('sliderError').value + ' %';
            }

            async function applyChaos() {
                const lat = parseInt(document.getElementById('sliderLatency').value);
                const err = parseInt(document.getElementById('sliderError').value) / 100.0;
                await fetch('/api/v1/chaos/configure', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({latency_ms: lat, error_rate: err, reset_circuit_breaker: false})
                });
                alert('Chaos parameters updated!');
            }

            async function resetCircuitBreaker() {
                await fetch('/api/v1/chaos/configure', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({latency_ms: 0, error_rate: 0.0, reset_circuit_breaker: true})
                });
                document.getElementById('sliderLatency').value = 0;
                document.getElementById('sliderError').value = 0;
                updateChaosUI();
                document.getElementById('circuitBadge').className = "px-3 py-1 text-xs font-semibold rounded-full bg-emerald-900 text-emerald-300 border border-emerald-700";
                document.getElementById('circuitBadge').innerText = "Circuit: HEALTHY";
                alert('Circuit Breaker Reset!');
            }

            async function runQuery() {
                const q = document.getElementById('queryInput').value;
                if(!q) return;

                const btn = document.getElementById('btnSubmit');
                btn.disabled = true;
                btn.innerText = "Retrieving & Synthesizing...";
                document.getElementById('outputArea').classList.add('hidden');
                document.getElementById('errorArea').classList.add('hidden');

                try {
                    const res = await fetch('/api/v1/query', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query: q})
                    });

                    if(!res.ok) {
                        const err = await res.json();
                        throw new Error(err.detail || 'Service Error');
                    }

                    const data = await res.json();
                    document.getElementById('answerText').innerText = data.answer || "No synthesis returned.";
                    document.getElementById('latencyBadge').innerText = data.total_latency_ms + ' ms';
                    
                    const docsList = document.getElementById('docsList');
                    docsList.innerHTML = '';
                    data.retrieved_docs.forEach(d => {
                        const sp = document.createElement('span');
                        sp.className = "px-2 py-0.5 text-xs bg-blue-900/60 border border-blue-600 rounded text-blue-200";
                        sp.innerText = d;
                        docsList.appendChild(sp);
                    });

                    document.getElementById('outputArea').classList.remove('hidden');
                } catch(e) {
                    const errArea = document.getElementById('errorArea');
                    errArea.innerText = e.message;
                    errArea.classList.remove('hidden');

                    if(e.message.includes('CIRCUIT BREAKER OPEN')) {
                        document.getElementById('circuitBadge').className = "px-3 py-1 text-xs font-semibold rounded-full bg-red-900 text-red-300 border border-red-700";
                        document.getElementById('circuitBadge').innerText = "Circuit: TRIPPED (OPEN)";
                    }
                } finally {
                    btn.disabled = false;
                    btn.innerText = "Execute Resilient Query";
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
