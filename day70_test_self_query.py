import requests

BASE_URL = "http://127.0.0.1:8000/api/v1/self-query"

def run_self_query_tests():
    print("=" * 80)
    print("DAY 70: SELF-QUERYING METADATA FILTERING TEST HARNESS")
    print("=" * 80)

    test_cases = [
        {
            "name": "Test 1: Triple Filter (Year=2026, Severity=P1, Region=EU)",
            "query": "Show me critical P1 database incident reports from 2026 in the EU region",
            "expected_id": "inc_2026_eu_p1",
            "rejected_ids": ["inc_2025_eu_p1", "inc_2026_us_p1", "inc_2026_eu_p3"]
        },
        {
            "name": "Test 2: Temporal Boundary Filter (Year=2025 Only)",
            "query": "What historical datacenter issues occurred in 2025 in EU?",
            "expected_id": "inc_2025_eu_p1",
            "rejected_ids": ["inc_2026_eu_p1"]
        }
    ]

    for tc in test_cases:
        print(f"\nRunning {tc['name']}...")
        print(f"Query: \"{tc['query']}\"")

        res = requests.post(BASE_URL, json={"query": tc["query"]})
        assert res.status_code == 200, f"Request failed: {res.text}"

        data = res.json()
        print(f"  • Extracted Semantic Query : \"{data['parsed_semantic_query']}\"")
        print(f"  • Injected Metadata Filters: {data['extracted_filters']}")
        print(f"  • Retrieved Record IDs     : {[item['id'] for item in data['retrieved_incidents']]}")
        print(f"  • Synthesized LLM Answer   : \"{data['answer']}\"")

        retrieved_ids = [item["id"] for item in data["retrieved_incidents"]]

        # Assert correct record is present
        assert tc["expected_id"] in retrieved_ids, (
            f"Expected {tc['expected_id']} in retrieved results: {retrieved_ids}"
        )

        # Assert filtered-out records are completely absent (no leakage)
        for rej in tc["rejected_ids"]:
            assert rej not in retrieved_ids, (
                f"Filter Leakage! Found forbidden record {rej} in results: {retrieved_ids}"
            )

        print("[PASSED] Self-querying engine isolated semantic intent and enforced metadata boundaries.")

    print("\n" + "=" * 80)
    print("All Self-Querying Metadata Filtering tests passed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    run_self_query_tests()
