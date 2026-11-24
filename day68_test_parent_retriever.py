import requests

BASE_URL = "http://127.0.0.1:8000/api/v1/parent-document-query"

def run_parent_retriever_tests():
    print("=" * 80)
    print("DAY 68: MULTI-VECTOR PARENT DOCUMENT RETRIEVER TEST HARNESS")
    print("=" * 80)

    test_cases = [
        {
            "name": "Test 1: Vault Certificate TTL Query",
            "question": "What is the default TTL for Vault intermediate dynamic certificates?",
            "expected_parent": "parent_vault_pki",
            "check": lambda ans: "72" in ans or "seventy-two" in ans.lower()
        },
        {
            "name": "Test 2: Kafka Broker Recovery Time Query",
            "question": "By how much does tiered storage reduce Kafka broker recovery time?",
            "expected_parent": "parent_kafka_tiered_storage",
            "check": lambda ans: "4" in ans or "four" in ans.lower()
        }
    ]

    for tc in test_cases:
        print(f"\nRunning {tc['name']}...")
        print(f"Query: \"{tc['question']}\"")

        res = requests.post(BASE_URL, json={"question": tc["question"], "top_k_children": 4})
        assert res.status_code == 200, f"Request failed: {res.text}"

        data = res.json()
        print(f"  • Matched Child Chunks      : {data['matched_children_count']}")
        print(f"  • Resolved Parent Documents : {data['resolved_parents_count']}")
        print(f"  • Parent Document IDs       : {data['resolved_parent_ids']}")
        print(f"  • Synthesized LLM Answer    : \"{data['answer']}\"")

        # 1. Verify parent resolution & deduplication
        assert tc["expected_parent"] in data["resolved_parent_ids"], (
            f"Expected {tc['expected_parent']}, got {data['resolved_parent_ids']}"
        )
        assert data["resolved_parents_count"] <= 2, "Parent document deduplication failed!"

        # 2. Verify answer contains ground truth fact
        assert tc["check"](data["answer"]), (
            f"Ground truth fact validation failed on answer: {data['answer']}"
        )

        print("[PASSED] Child vector search accurately retrieved and expanded full Parent Document.")

    print("\n" + "=" * 80)
    print("All Multi-Vector Parent Document Retriever tests completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    run_parent_retriever_tests()
