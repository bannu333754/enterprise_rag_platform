import requests

BASE_URL = "http://127.0.0.1:8000/api/v1/contextual-query"

def run_compression_benchmarks():
    print("=" * 80)
    print("DAY 67: CONTEXTUAL COMPRESSION & SENTENCE-WINDOW BENCHMARK HARNESS")
    print("=" * 80)

    test_queries = [
        {
            "name": "Benchmark 1: CoreDNS Specific Memory Tuning",
            "question": "What memory limit should be configured for CoreDNS?",
            "check": lambda ans: "170" in ans
        },
        {
            "name": "Benchmark 2: Redis Cluster Hash Slot Partitioning",
            "question": "How many virtual hash slots exist in a Redis cluster?",
            "check": lambda ans: "16,384" in ans or "16384" in ans.replace(",", "")
        }
    ]

    for tc in test_queries:
        print(f"\nRunning {tc['name']}...")
        print(f"Query: \"{tc['question']}\"")

        # Test A: Raw Window
        res_raw = requests.post(BASE_URL, json={"question": tc["question"], "enable_compression": False})
        data_raw = res_raw.json()

        # Test B: Compressed
        res_comp = requests.post(BASE_URL, json={"question": tc["question"], "enable_compression": True})
        data_comp = res_comp.json()

        print(f"  • Raw Window Context Tokens   : {data_raw['raw_context_tokens']} tokens")
        print(f"  • Compressed Context Tokens   : {data_comp['compressed_context_tokens']} tokens")
        print(f"  • Token Reduction / Savings   : {data_comp['compression_ratio_percent']}%")
        print(f"  • Filtered Context Extracted  : \"{data_comp['final_context_used']}\"")
        print(f"  • Synthesized LLM Answer      : \"{data_comp['answer']}\"")

        assert data_comp["compression_ratio_percent"] >= 30.0, "Expected at least 30% contextual compression!"
        assert tc["check"](data_comp["answer"]), f"Answer validation failed on output: {data_comp['answer']}"

        print("[PASSED] Compression pruned noise while preserving factual accuracy.")

    print("\n" + "=" * 80)
    print("All Contextual Compression & Sentence-Window tests passed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    run_compression_benchmarks()
