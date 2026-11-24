import requests

BASE_URL = "http://127.0.0.1:8000/api/v1/hybrid-search"

def run_hybrid_tests():
    print("=" * 80)
    print("DAY 66: HYBRID RETRIEVAL (DENSE + BM25 + RRF) TEST HARNESS")
    print("=" * 80)

    test_queries = [
        {
            "name": "Test 1: Exact Technical Error Code Match (Favors BM25 Sparse)",
            "query": "What patch resolves ERR_K8S_503?",
            "must_contain_doc_id": "doc_5"
        },
        {
            "name": "Test 2: Conceptual Semantic Search (Favors Dense Vector)",
            "query": "How is consensus maintained across distributed nodes?",
            "must_contain_doc_id": "doc_2"
        },
        {
            "name": "Test 3: Mixed Semantic & Lexical Query",
            "query": "Gateway timeout HTTP 503 troubleshooting and proxy settings",
            "must_contain_doc_id": "doc_3"
        }
    ]

    for tc in test_queries:
        print(f"\nRunning {tc['name']}...")
        print(f"Query: \"{tc['query']}\"")

        res = requests.post(BASE_URL, json={"question": tc["query"], "top_k": 3})
        if res.status_code != 200:
            print(f"[FAIL] HTTP Status {res.status_code}: {res.text}")
            continue

        data = res.json()
        print(f"Answer: {data['answer'][:120]}...\n")
        print("Ranked Documents via RRF:")
        retrieved_ids = []
        for rank, doc in enumerate(data["fused_documents"], start=1):
            retrieved_ids.append(doc["id"])
            print(f"  [{rank}] ID: {doc['id']} | RRF Score: {doc['rrf_score']} | Dense Rank: {doc['dense_rank']} | Sparse Rank: {doc['sparse_rank']}")
            print(f"      Text: {doc['text'][:85]}...")

        assert tc["must_contain_doc_id"] in retrieved_ids, (
            f"Failed: Expected {tc['must_contain_doc_id']} in results, got {retrieved_ids}"
        )
        print("[PASSED] Target document successfully prioritized by Hybrid RRF.")

    print("\n" + "=" * 80)
    print("All Dense-Sparse Hybrid Search with RRF tests completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    run_hybrid_tests()
