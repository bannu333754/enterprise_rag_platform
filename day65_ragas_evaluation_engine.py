import time
import json
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
import chromadb
from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [EVAL] %(message)s")
logger = logging.getLogger("RagasEvaluator")

# Initialize models and clients
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key="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")

# ----------------- Evaluation Data Schemas -----------------
class EvalSample(BaseModel):
    question: str
    ground_truth: str
    contexts: List[str]
    answer: Optional[str] = None

class MetricScores(BaseModel):
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    answer_relevance: float = Field(..., ge=0.0, le=1.0)
    context_precision: float = Field(..., ge=0.0, le=1.0)
    composite_ragas_score: float = Field(..., ge=0.0, le=1.0)
    passed_gate: bool

# ----------------- LLM-as-a-Judge Scorers -----------------
class RagasEvaluator:
    def __init__(self, client: Groq, model: str = "openai/gpt-oss-20b"):
        self.client = client
        self.model = model

    def evaluate_faithfulness(self, question: str, answer: str, contexts: List[str]) -> float:
        """
        LLM Judge: Extracts claims from the answer and checks if each claim
        is strictly supported by the provided context chunks.
        """
        context_str = "\n".join(contexts)
        prompt = f"""You are an unbiased automated RAG judge evaluating faithfulness.
Task: Given the Context and the Candidate Answer, determine what fraction of factual claims in the Answer can be directly deduced from the Context.

Context:
{context_str}

Candidate Answer:
{answer}

Output format: Return ONLY a valid JSON object with keys "total_claims" (integer), "supported_claims" (integer), and "score" (float between 0.0 and 1.0).
JSON:"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150
            )
            raw = resp.choices[0].message.content.strip()
            # Clean possible markdown wrapping
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            parsed = json.loads(raw)
            return float(parsed.get("score", 0.0))
        except Exception as e:
            logger.warning(f"Faithfulness LLM scoring fallback triggered: {e}")
            # Semantic fallback check via cosine similarity
            ans_vec = embed_model.encode(answer)
            ctx_vec = embed_model.encode(context_str)
            sim = float(util.cos_sim(ans_vec, ctx_vec)[0][0])
            return min(1.0, max(0.0, sim))

    def evaluate_answer_relevance(self, question: str, answer: str) -> float:
        """
        Calculates semantic alignment between original user query and candidate answer.
        """
        q_vec = embed_model.encode(question)
        ans_vec = embed_model.encode(answer)
        score = float(util.cos_sim(q_vec, ans_vec)[0][0])
        return round(min(1.0, max(0.0, score)), 4)

    def evaluate_context_precision(self, ground_truth: str, contexts: List[str]) -> float:
        """
        Computes precision based on whether high-ranking chunks contain relevant ground-truth facts.
        """
        if not contexts:
            return 0.0

        gt_vec = embed_model.encode(ground_truth)
        hits = []
        for ctx in contexts:
            c_vec = embed_model.encode(ctx)
            sim = float(util.cos_sim(gt_vec, c_vec)[0][0])
            hits.append(1 if sim >= 0.55 else 0)

        # Average precision formula
        score = sum(hits) / len(hits) if hits else 0.0
        return round(score, 4)

    def run_full_evaluation(self, sample: EvalSample, candidate_answer: str) -> MetricScores:
        f_score = self.evaluate_faithfulness(sample.question, candidate_answer, sample.contexts)
        r_score = self.evaluate_answer_relevance(sample.question, candidate_answer)
        p_score = self.evaluate_context_precision(sample.ground_truth, sample.contexts)

        # Harmonic/Composite Ragas score
        composite = round((f_score * 0.4) + (r_score * 0.4) + (p_score * 0.2), 4)

        # CI/CD Quality Gate Thresholds
        passed = (f_score >= 0.70) and (r_score >= 0.65) and (p_score >= 0.50)

        return MetricScores(
            faithfulness=round(f_score, 4),
            answer_relevance=round(r_score, 4),
            context_precision=round(p_score, 4),
            composite_ragas_score=composite,
            passed_gate=passed
        )

evaluator = RagasEvaluator(client=groq_client)
