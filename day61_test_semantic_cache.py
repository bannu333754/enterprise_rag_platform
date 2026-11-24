import requests
import time

BASE_URL = "http://127.0.0.1:8000"

def test_cache_pipeline():
    print("=" * 75)
    print("DAY 61: SEMANTIC CACHING & LATENCY BENCHMARK HARNESS")
    print("=" * 75)

    test_queries = [
        # Query 1: Initial cold run -> Expect CACHE_MISS
        {
            "query": "What is Redis used for in infrastructure?",
            "expected_status": "CACHE_MISS"
        },
        # Query 2: Exact identical run -> Expect EXACT_HIT (Sub-10ms)
        {
            "query": "What is Redis used for in infrastructure?",
            "expected_status": "EXACT_HIT"
        },
        # Query 3: Semantically equivalent variation -> Expect SEMANTIC_HIT
        {
            "query": "Can you explain what Redis is used for in systems?",
            "expected_status": "SEMANTIC_HIT"
        },
        # Query 4: Unrelated query -> Expect CACHE_MISS
        {
            "query": "How is cosine similarity calculated?",
            "expected_status": "CACHE_MISS"
        },
        # Query 5: Semantically equivalent variation of Query 4 -> Expect SEMANTIC_HIT
        {
            "query": "What does cosine similarity measure between two vectors?",
            "expected_status": "SEMANTIC_HIT"
        }
    ]

    for idx, t in enumerate(test_queries, start=1):
        print(f"\n[Test {idx}] Query: \"{t['query']}\"")
        start = time.perf_counter()
        res = requests.post(
            f"{BASE_URL}/api/v1/cached-query",
            json={"question": t["query"]}
        )
        wall_time_ms = (time.perf_counter() - start) * 1000

        if res.status_code != 200:
            print(f"[FAIL] HTTP {res.status_code}: {res.text}")
            continue

        data = res.json()
        print(f"  -> Cache Status   : {data['cache_status']} (Expected: {t['expected_status']})")
        print(f"  -> Pipeline Time  : {data['latency_ms']} ms (Wall Time: {wall_time_ms:.2f} ms)")
        print(f"  -> Match Score    : {data['similarity_score']}")
        print(f"  -> Tokens Saved   : {data['tokens_saved']}")
        print(f"  -> Answer Snippet : {data['answer'][:80]}...")

        assert data["cache_status"] == t["expected_status"], f"Expected {t['expected_status']} but got {data['cache_status']}"

    # Verify cache statistics endpoint
    print("\n" + "-" * 75)
    print("Retrieving In-Memory Cache Telemetry Statistics...")
    stats_res = requests.get(f"{BASE_URL}/api/v1/cache/stats")
    stats = stats_res.json()
    print(f"Active Entries in Cache : {stats['total_entries']}")
    print(f"Exact Hash Keys Cached  : {stats['exact_keys']}")
    for item in stats["active_items"]:
        print(f"  * Query: '{item['query']}' | Hits: {item['hits']} | Age: {item['age_sec']}s")

    print("\n" + "=" * 75)
    print("All Semantic Caching & Latency tests passed successfully!")
    print("=" * 75)

if __name__ == "__main__":
    test_cache_pipeline()
