import requests
import time

BASE_URL = "http://127.0.0.1:8000"

def run_rate_limiting_tests():
    print("=" * 75)
    print("DAY 62: ENTERPRISE TOKEN-BUCKET RATE LIMITING & DDOS TEST HARNESS")
    print("=" * 75)

    free_headers = {"X-API-Key": "key_free_user", "Content-Type": "application/json"}
    body = {"question": "What is the token bucket algorithm?"}

    print("\n1. Simulating burst requests on Free Tier (Capacity = 5 tokens)...")
    rejected_count = 0
    accepted_count = 0

    for i in range(1, 9):
        res = requests.post(f"{BASE_URL}/api/v1/throttled-query", json=body, headers=free_headers)
        rem = res.headers.get("X-RateLimit-Remaining", "0")
        
        if res.status_code == 200:
            accepted_count += 1
            print(f"  Request {i}: [200 OK] Remaining Allowance: {rem}")
        elif res.status_code == 429:
            rejected_count += 1
            retry_after = res.headers.get("Retry-After", "Unknown")
            print(f"  Request {i}: [429 Too Many Requests] Retry-After: {retry_after}s")

    print(f"\nBurst Test Summary: Accepted={accepted_count}, Blocked={rejected_count}")
    assert accepted_count == 5, f"Expected 5 accepted burst requests, got {accepted_count}"
    assert rejected_count == 3, f"Expected 3 rejected requests, got {rejected_count}"

    print("\n2. Testing Enterprise Tier Client (Capacity = 30 tokens)...")
    ent_headers = {"X-API-Key": "key_enterprise_corp", "Content-Type": "application/json"}
    res_ent = requests.post(f"{BASE_URL}/api/v1/throttled-query", json=body, headers=ent_headers)
    print(f"  Enterprise Request: [{res_ent.status_code}] Remaining: {res_ent.headers.get('X-RateLimit-Remaining')}")
    assert res_ent.status_code == 200, "Enterprise tier should succeed under load"

    print("\n3. Testing Token Bucket Refill (Waiting 5 seconds for 1 free token)...")
    time.sleep(5.2)
    refill_res = requests.post(f"{BASE_URL}/api/v1/throttled-query", json=body, headers=free_headers)
    print(f"  Post-Cooldown Request: [{refill_res.status_code}] Remaining: {refill_res.headers.get('X-RateLimit-Remaining')}")
    assert refill_res.status_code == 200, "Refill failed to grant new allowance"

    print("\n" + "=" * 75)
    print("All Rate Limiting, HTTP 429 Throttling, and Cooldown tests passed!")
    print("=" * 75)

if __name__ == "__main__":
    run_rate_limiting_tests()
