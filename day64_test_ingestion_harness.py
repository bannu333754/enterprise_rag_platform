import time
import requests

BASE_URL = "http://127.0.0.1:8000"

def run_async_ingestion_tests():
    print("=" * 75)
    print("DAY 64: ASYNCHRONOUS INGESTION, CHUNKING & WEBHOOK HARNESS")
    print("=" * 75)

    sample_doc = (
        "Enterprise cloud migration requires zero-trust network topology and continuous observability. "
        "Kubernetes orchestration schedules microservices with horizontal auto-scaling and pod disruption budgets. "
        "Prometheus scrapes runtime metrics every five seconds to capture latency anomalies and CPU throttling. "
        "Distributed tracing propagates W3C traceparent headers across every API gateway boundary. "
        "Semantic caching prevents redundant vector searches by resolving high-confidence cosine matches in under ten milliseconds."
    )

    payload = {
        "filename": "cloud_architecture_standard_v2.txt",
        "content": sample_doc * 4,  # Multiply to generate multi-chunk payload
        "webhook_url": f"{BASE_URL}/api/v1/mock-webhook-receiver"
    }

    # 1. Submit Ingestion Job
    print("\n1. Submitting asynchronous ingestion job...")
    res = requests.post(f"{BASE_URL}/api/v1/ingest/async", json=payload)
    assert res.status_code == 202, f"Expected HTTP 202, got {res.status_code}"
    
    data = res.json()
    job_id = data["job_id"]
    print(f"  -> Job Accepted: {job_id}")
    print(f"  -> Initial Status: {data['status']}")
    print(f"  -> Poll Endpoint: {data['poll_endpoint']}")

    # 2. Poll Status State Machine
    print("\n2. Polling job state until completion...")
    completed = False
    for i in range(15):
        poll = requests.get(f"{BASE_URL}/api/v1/ingest/status/{job_id}").json()
        status = poll["status"]
        progress = poll["progress_percentage"]
        chunks = poll["total_chunks"]
        print(f"  Poll [{i+1:02d}]: Status={status:<10} | Progress={progress}% | Chunks={chunks}")

        if status == "INDEXED":
            completed = True
            break
        time.sleep(0.4)

    assert completed, "Ingestion job timed out or failed to reach INDEXED status!"

    # 3. Verify Webhook Dispatch
    print("\n3. Verifying webhook dispatch to client receiver...")
    time.sleep(0.5)
    hook_res = requests.get(f"{BASE_URL}/api/v1/mock-webhook-receiver/logs").json()
    print(f"  Total webhooks received: {hook_res['count']}")
    
    matching_hooks = [h for h in hook_res["logs"] if h.get("job_id") == job_id]
    assert len(matching_hooks) > 0, "Webhook was not successfully delivered!"
    print(f"  -> Webhook payload confirmed for job: {matching_hooks[0]['job_id']}")
    print(f"  -> Indexed chunk count: {matching_hooks[0]['total_chunks']}")

    print("\n" + "=" * 75)
    print("All Asynchronous Ingestion, Chunking & Webhook tests passed!")
    print("=" * 75)

if __name__ == "__main__":
    run_async_ingestion_tests()
