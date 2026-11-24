import json
from day65_ragas_evaluation_engine import RagasEvaluator, EvalSample, groq_client

def run_evaluation_suite():
    print("=" * 80)
    print("DAY 65: ENTERPRISE RAGAS BENCHMARK & LLM-AS-A-JUDGE QUALITY GATE")
    print("=" * 80)

    evaluator = RagasEvaluator(client=groq_client)

    # Standardized evaluation benchmark dataset
    test_cases = [
        {
            "name": "Benchmark 1: High-Quality Grounded Retrieval",
            "question": "What is the primary role of Celery in asynchronous systems?",
            "ground_truth": "Celery executes background asynchronous task queues decoupled from the main web server loop.",
            "contexts": [
                "Celery is an asynchronous distributed task queue that processes jobs off the primary HTTP thread.",
                "Redis acts as the message broker passing payloads between web API nodes and Celery worker daemons."
            ],
            "candidate_answer": "Celery functions as a distributed background task queue that executes heavy jobs asynchronously away from the main web loop."
        },
        {
            "name": "Benchmark 2: Hallucination / Ungrounded Claim Test",
            "question": "What is the memory limit of a ChromaDB collection?",
            "ground_truth": "ChromaDB memory consumption depends on vector dimensions and index size in RAM or SQLite storage.",
            "contexts": [
                "ChromaDB stores embeddings in memory and persists vector collections directly to local disk or SQLite."
            ],
            # Deliberately ungrounded answer with fabricated facts
            "candidate_answer": "ChromaDB strictly limits all collections to 256 megabytes and forces automated server reboots every Sunday."
        }
    ]

    all_results = []

    for idx, tc in enumerate(test_cases, start=1):
        print(f"\n[{idx}] Evaluating: {tc['name']}")
        sample = EvalSample(
            question=tc["question"],
            ground_truth=tc["ground_truth"],
            contexts=tc["contexts"]
        )

        scores = evaluator.run_full_evaluation(sample, tc["candidate_answer"])
        all_results.append({"name": tc["name"], "scores": scores.dict()})

        print(f"  • Faithfulness / Groundedness : {scores.faithfulness}")
        print(f"  • Answer Relevance            : {scores.answer_relevance}")
        print(f"  • Context Precision           : {scores.context_precision}")
        print(f"  • Composite Ragas Score       : {scores.composite_ragas_score}")
        print(f"  • Quality Gate Status         : {'[PASSED]' if scores.passed_gate else '[FAILED / REGRESSION]'}")

    print("\n" + "=" * 80)
    print("CI/CD QUALITY GATE VALIDATION SUMMARY")
    print("=" * 80)

    # Validate that high-quality sample passes gate
    assert all_results[0]["scores"]["passed_gate"] is True, "Benchmark 1 should pass quality gate!"
    # Validate that hallucination sample fails gate
    assert all_results[1]["scores"]["faithfulness"] < 0.70 or all_results[1]["scores"]["passed_gate"] is False, "Hallucinated answer must fail gate!"

    print("Quality Gates successfully flagged grounded vs hallucinated responses!")
    print("Generated evaluation metrics report saved to ragas_eval_report.json")

    with open("ragas_eval_report.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    run_evaluation_suite()
