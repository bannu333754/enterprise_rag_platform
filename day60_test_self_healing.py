import requests
import time

BASE_URL = "http://127.0.0.1:8000"

def run_self_healing_tests():
    print("=" * 70)
    print("DAY 60: AUTONOMOUS SELF-HEALING & CIRCUIT BREAKER TEST HARNESS")
    print("=" * 70)

    # 1. Normal Query Check
    print("\n1. Testing standard grounded query...")
    r = requests.post(f"{BASE_URL}/api/v1/resilient-query", json={"question": "How does HPA scale?"})
    print(f"Status: {r.status_code} | Body: {r.json()['status']} | Answer: {r.json()['answer']}")
    assert r.status_code == 200

    # 2. Trigger Circuit Breaker Trip
    print("\n2. Simulating 3 consecutive failures to trip Circuit Breaker...")
    for i in range(3):
        res = requests.post(f"{BASE_URL}/api/v1/resilient-query", json={"question": "Fault trigger", "simulate_failure": True})
        print(f"Call {i+1} -> State: {res.json()['circuit_state']}")

    # 3. Verify Fast-Failing when OPEN
    print("\n3. Testing immediate fallback during OPEN circuit state...")
    open_res = requests.post(f"{BASE_URL}/api/v1/resilient-query", json={"question": "Normal query during trip"})
    print(f"Response: {open_res.json()['status']} | Circuit State: {open_res.json()['circuit_state']}")
    assert open_res.json()["status"] == "FALLBACK_CIRCUIT_OPEN"

    # 4. Verify Dead-Letter Queue Ingestion
    print("\n4. Checking Dead-Letter Queue (DLQ) records...")
    dlq_res = requests.get(f"{BASE_URL}/api/v1/dlq")
    dlq_data = dlq_res.json()
    print(f"Total entries in DLQ: {dlq_data['dlq_size']}")
    for idx, item in enumerate(dlq_data['records']):
        print(f"  [{idx+1}] Reason: {item['reason']} | Query: {item['query']}")

    print("\n" + "=" * 70)
    print("All Self-Healing & Circuit Breaker verifications completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    run_self_healing_tests()
