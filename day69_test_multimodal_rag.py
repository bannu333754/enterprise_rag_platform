import requests

BASE_URL = "http://127.0.0.1:8000/api/v1/multimodal-query"

def run_multimodal_tests():
    print("=" * 80)
    print("DAY 69: MULTI-MODAL RAG (TABLES + DIAGRAMS + TEXT) TEST HARNESS")
    print("=" * 80)

    test_cases = [
        {
            "name": "Test 1: Tabular Financial Query (Table Extraction)",
            "question": "What was the EPS and Revenue reported in Q3 2026?",
            "filter_type": None,
            "expected_type": "table",
            "check": lambda ans: "3.12" in ans and ("5,600" in ans or "5600" in ans.replace(",", ""))
        },
        {
            "name": "Test 2: Architecture Diagram Query (Visual Blueprint)",
            "question": "Which component verifies SPIFFE identity certificates in the zero-trust architecture?",
            "filter_type": "image_diagram",
            "expected_type": "image_diagram",
            "check": lambda ans: "envoy" in ans.lower()
        }
    ]

    for tc in test_cases:
        print(f"\nRunning {tc['name']}...")
        print(f"Query: \"{tc['question']}\"")

        payload = {"question": tc["question"], "filter_type": tc["filter_type"]}
        res = requests.post(BASE_URL, json=payload)
        assert res.status_code == 200, f"Request failed: {res.text}"

        data = res.json()
        print(f"  • Detected Modality Types : {data['detected_types']}")
        print(f"  • Retrieved Chunks Count  : {len(data['retrieved_items'])}")
        print(f"  • Synthesized LLM Answer  : \"{data['answer']}\"")

        # Verify modality matching
        assert tc["expected_type"] in data["detected_types"], (
            f"Expected {tc['expected_type']} in {data['detected_types']}"
        )

        # Verify ground truth
        assert tc["check"](data["answer"]), f"Answer assertion failed: {data['answer']}"

        print("[PASSED] Multi-modal retrieval and synthesis verified successfully.")

    print("\n" + "=" * 80)
    print("All Multi-Modal RAG tests completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    run_multimodal_tests()
