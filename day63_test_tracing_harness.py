import requests
import json

BASE_URL = "http://127.0.0.1:8000/api/v1/traced-query"

def run_tracing_tests():
    print("=" * 75)
    print("DAY 63: OPENTELEMETRY DISTRIBUTED TRACING & SPAN HARNESS")
    print("=" * 75)

    # Synthetic incoming W3C traceparent (version 00, custom trace_id, custom parent_span_id, sampled 01)
    synthetic_trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    synthetic_parent_span_id = "00f067aa0ba902b7"
    w3c_traceparent = f"00-{synthetic_trace_id}-{synthetic_parent_span_id}-01"

    headers = {
        "Content-Type": "application/json",
        "traceparent": w3c_traceparent
    }
    payload = {"question": "How does distributed tracing track microservice requests?"}

    print(f"\n1. Sending request with injected W3C traceparent:\n   Header: {w3c_traceparent}")
    res = requests.post(BASE_URL, json=payload, headers=headers)

    if res.status_code != 200:
        print(f"[FAIL] HTTP {res.status_code}: {res.text}")
        return

    data = res.json()
    telemetry = data["telemetry"]
    returned_traceparent = res.headers.get("traceparent", "")

    print(f"\n2. Verifying Trace Context Propagation:")
    print(f"  -> Extracted Trace ID     : {telemetry['trace_id']}")
    print(f"  -> Returned Traceparent   : {returned_traceparent}")
    print(f"  -> Root Span ID           : {telemetry['root_span_id']}")
    print(f"\n3. Latency Breakdown per Span:")
    print(f"  -> [embed.generate]       : {telemetry['embed_ms']} ms")
    print(f"  -> [vector.retrieval]     : {telemetry['db_ms']} ms")
    print(f"  -> [llm.synthesis]        : {telemetry['llm_ms']} ms")
    print(f"  -> [total_pipeline]       : {telemetry['total_ms']} ms")
    print(f"\n4. Synthesized Output:\n  {data['answer']}")

    # Validation checks
    assert telemetry["trace_id"] == synthetic_trace_id, "Trace ID mismatch! Context propagation failed."
    assert "traceparent" in res.headers, "Outgoing response missing traceparent header."
    assert telemetry["embed_ms"] > 0 and telemetry["db_ms"] > 0 and telemetry["llm_ms"] > 0

    print("\n" + "=" * 75)
    print("All OpenTelemetry Distributed Tracing & Span tests passed successfully!")
    print("=" * 75)

if __name__ == "__main__":
    run_tracing_tests()
