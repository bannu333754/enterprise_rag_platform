import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 41 Federated Multi-Tenant RAG", layout="wide")

st.title("🛡️ Day 41: Federated Multi-Tenant Enterprise RAG")
st.caption("Cryptographic Tenant Isolation & Fine-Grained Document Clearance Filtering (ACL)")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("🏢 Multi-Tenant Ingestion")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Reset Session"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    upload_tenant = st.text_input("Tenant Organization ID", value="CorpAlpha")
    upload_clearance = st.selectbox(
        "Document Clearance Level",
        options=["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"],
        index=1
    )
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

    if st.button("Index with ACL Tags"):
        if uploaded_file and upload_tenant.strip():
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            data = {
                "tenant_id": upload_tenant.strip(),
                "clearance_tier": upload_clearance
            }
            try:
                res = requests.post(f"{API_URL}/upload", files=files, data=data)
                if res.status_code == 200:
                    r_json = res.json()
                    st.success(f"Indexed {r_json.get('chunks_indexed')} chunks for [{upload_tenant}] as [{upload_clearance}]!")
                    if uploaded_file.name not in st.session_state.uploaded_files_list:
                        st.session_state.uploaded_files_list.append(uploaded_file.name)
                else:
                    st.error(f"Upload failed: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
        else:
            st.warning("Enter a Tenant ID and select a PDF.")

    st.divider()
    st.header("👤 Active User Context")
    active_tenant = st.text_input("Active Tenant Context", value="CorpAlpha")
    active_user_id = st.text_input("User ID", value="emp_1042")
    active_clearance = st.selectbox(
        "User Security Clearance",
        options=["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"],
        index=0
    )

    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Filter Document Source", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Query enterprise knowledge (testing cross-tenant or clearance boundaries)...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "tenant_id": active_tenant.strip(),
        "user_id": active_user_id.strip(),
        "clearance_role": active_clearance,
        "question": user_query,
        "target_doc": selected_doc
    }

    with st.chat_message("assistant"):
        with st.spinner("Enforcing tenant boundary and filtering security clearance tags..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("final_answer", "")
                    applied_filter = data.get("applied_acl_filter", "{}")
                    evidence = data.get("accessible_evidence", [])

                    st.markdown(answer)

                    # Security Audit Panel
                    st.markdown("### 🔐 Security & ACL Audit Trace")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Tenant Boundary", data.get("tenant_id"))
                    with c2:
                        st.metric("Caller Clearance", data.get("user_clearance"))
                    with c3:
                        st.metric("Accessible Passages", len(evidence))

                    st.code(f"Dynamic Metadata Filter: {applied_filter}", language="json")

                    if evidence:
                        with st.expander(f"📑 Authorized Passages Surfaced ({len(evidence)})"):
                            for i, it in enumerate(evidence):
                                st.markdown(f"**Doc {i+1}** | `{it['source']}` (p.{it['page']}) | Required Tier: `{it['required_clearance']}` | Score: `{it['score']}`")
                                st.write(f"\"{it['text']}\"")
                                st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
