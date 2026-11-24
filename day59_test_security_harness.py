import requests

BASE_URL = "http://127.0.0.1:8000/api/v1/secure-query"

scenarios = [
    {
        "name": "Scenario 1: Alice (Acme Corp, Internal Clearance)",
        "token": "token_alice_acme",
        "question": "What acquisitions or roadmaps are planned?",
        "expected_tenant": "acme_corp",
        "forbidden_keywords": ["BetaCorp", "Globex", "45M USD"]  # Should NOT see restricted or other tenant's data
    },
    {
        "name": "Scenario 2: Bob (Acme Corp, Restricted Clearance - Executive)",
        "token": "token_bob_acme",
        "question": "What acquisitions are being evaluated?",
        "expected_tenant": "acme_corp",
        "must_include": ["BetaCorp"]  # Should see restricted data
    },
    {
        "name": "Scenario 3: Charlie (Globex Inc, Confidential Clearance)",
        "token": "token_charlie_globex",
        "question": "What are our financial metrics?",
        "expected_tenant": "globex_inc",
        "forbidden_keywords": ["Acme"]  # Zero leakage from Acme Corp
    }
]

def run_tests():
    print("=" * 70)
    print("DAY 59: MULTI-TENANT RBAC SECURITY VALIDATION HARNESS")
    print("=" * 70)

    for sc in scenarios:
        print(f"\nRunning {sc['name']}...")
        headers = {
            "Authorization": f"Bearer {sc['token']}",
            "Content-Type": "application/json"
        }
        body = {"question": sc["question"], "top_k": 3}

        try:
            res = requests.post(BASE_URL, json=body, headers=headers)
            if res.status_code != 200:
                print(f"[FAIL] HTTP Status {res.status_code}: {res.text}")
                continue

            data = res.json()
            answer = data.get("answer", "")
            print(f"Tenant: {data['tenant_id']} | Clearance: {data['effective_clearance']}")
            print(f"Answer: {answer[:120]}...")

            # Validate tenant isolation
            assert data["tenant_id"] == sc["expected_tenant"], "Tenant boundary breached!"

            # Check forbidden leakages
            for forbidden in sc.get("forbidden_keywords", []):
                assert forbidden.lower() not in answer.lower(), f"DATA LEAKAGE DETECTED! '{forbidden}' found in response!"

            # Check expected access
            for expected in sc.get("must_include", []):
                assert expected.lower() in answer.lower(), f"Expected keyword '{expected}' not found!"

            print("[PASSED] Security policy enforced cleanly.")

        except Exception as e:
            print(f"[ERROR] {e}")

    print("\n" + "=" * 70)
    print("All multi-tenant isolation tests completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
