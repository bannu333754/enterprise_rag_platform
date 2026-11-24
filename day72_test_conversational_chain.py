import requests

BASE_URL = "http://127.0.0.1:8000/api/v1/chat"

def run_conversational_tests():
    print("=" * 80)
    print("DAY 72: MULTI-TURN CONVERSATIONAL RAG & SESSION MEMORY TEST HARNESS")
    print("=" * 80)

    session_id = "test_sess_infra_99"

    # Turn 1: Explicit introduction of entity
    t1_q = "What is the primary role of Envoy Gateway in the infrastructure?"
    print(f"\n[Turn 1] User: \"{t1_q}\"")
    r1 = requests.post(BASE_URL, json={"session_id": session_id, "question": t1_q})
    assert r1.status_code == 200, f"Turn 1 failed: {r1.text}"
    d1 = r1.json()
    print(f"         Condensed Query : \"{d1['condensed_query']}\"")
    print(f"         Retrieved Doc   : {d1['retrieved_chunk_id']}")
    print(f"         Assistant Answer: \"{d1['answer']}\"")
    assert d1["retrieved_chunk_id"] == "doc_envoy_gateway"

    # Turn 2: Follow-up question using pronoun "it" and missing entity name
    t2_q = "What port does it listen on for mTLS traffic?"
    print(f"\n[Turn 2] User: \"{t2_q}\"")
    r2 = requests.post(BASE_URL, json={"session_id": session_id, "question": t2_q})
    assert r2.status_code == 200, f"Turn 2 failed: {r2.text}"
    d2 = r2.json()
    print(f"         Condensed Query : \"{d2['condensed_query']}\"")
    print(f"         Retrieved Doc   : {d2['retrieved_chunk_id']}")
    print(f"         Assistant Answer: \"{d2['answer']}\"")
    
    # Assert query condensation resolved pronoun "it" to "Envoy Gateway"
    assert "envoy" in d2["condensed_query"].lower(), (
        f"Query condensation failed to resolve pronoun reference: {d2['condensed_query']}"
    )
    assert "8443" in d2["answer"], f"Expected port 8443 in answer: {d2['answer']}"
    assert d2["retrieved_chunk_id"] == "doc_envoy_gateway"

    # Turn 3: Second follow-up regarding rate-limiting limits
    t3_q = "What is its default rate limit threshold?"
    print(f"\n[Turn 3] User: \"{t3_q}\"")
    r3 = requests.post(BASE_URL, json={"session_id": session_id, "question": t3_q})
    assert r3.status_code == 200, f"Turn 3 failed: {r3.text}"
    d3 = r3.json()
    print(f"         Condensed Query : \"{d3['condensed_query']}\"")
    print(f"         Retrieved Doc   : {d3['retrieved_chunk_id']}")
    print(f"         Assistant Answer: \"{d3['answer']}\"")

    ans_normalized = d3["answer"].replace(",", "")
    assert "2500" in ans_normalized or "2,500" in d3["answer"], (
        f"Expected 2,500 rate limit in answer: {d3['answer']}"
    )
    assert d3["retrieved_chunk_id"] == "doc_envoy_rate_limit"

    print("\n" + "=" * 80)
    print("All Multi-Turn Conversational Session Memory tests passed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    run_conversational_tests()
