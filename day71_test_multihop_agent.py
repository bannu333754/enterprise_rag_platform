import requests

BASE_URL = "http://127.0.0.1:8000/api/v1/multihop-query"

def run_multihop_tests():
    print("=" * 80)
    print("DAY 71: AGENTIC MULTI-HOP RAG TEST HARNESS")
    print("=" * 80)

    # Multi-Hop Question:
    # Hop 1 must uncover: Incident INC-9042 -> failed component is NebulaDB
    # Hop 2 must uncover: NebulaDB -> write throughput is 1,200 IOPS
    query = "What is the maximum write throughput of the storage engine that caused incident INC-9042?"

    print(f"\nExecuting Multi-Hop Agent Query:\n\"{query}\"")
    res = requests.post(BASE_URL, json={"query": query, "max_hops": 3})
    assert res.status_code == 200, f"Request failed: {res.text}"

    data = res.json()
    print(f"\n• Total Hops Executed: {data['total_hops_executed']}")
    print(f"• Synthesized Answer: \"{data['final_answer']}\"\n")

    print("Execution Trace Across Graph Nodes:")
    for hop_info in data["hops_detail"]:
        print(f"  [Hop {hop_info['hop']}] Query: \"{hop_info['searched_query']}\"")
        print(f"         Retrieved: {hop_info['retrieved_docs']}")
        print(f"         Snippet  : {hop_info['evidence'][0][:80]}...")

    # Verifications
    assert data["total_hops_executed"] >= 2, "Failed: Expected multi-hop execution (>= 2 hops)!"
    
    # Check that both dependent documents were retrieved in the multi-hop trail
    all_retrieved = []
    for h in data["hops_detail"]:
        all_retrieved.extend(h["retrieved_docs"])
    
    assert "doc_incident_alpha" in all_retrieved, "Failed to retrieve initial incident report in Hop 1"
    assert "doc_nebuladb_specs" in all_retrieved, "Failed to retrieve secondary spec document in Hop 2"

    ans_str = data["final_answer"].replace(",", "")
    assert "1200" in ans_str or "1,200" in data["final_answer"], "Answer missed required fact: 1,200 IOPS"

    print("\n[PASSED] Multi-hop graph agent traversed dependent document nodes and resolved the query.")
    print("=" * 80)

if __name__ == "__main__":
    run_multihop_tests()
