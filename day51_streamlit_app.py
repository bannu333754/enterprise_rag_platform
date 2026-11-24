import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 51 Tool-Augmented RAG", layout="wide")

st.title("🛠️ Day 51: Tool-Augmented & Schema-Enforced Agentic RAG")
st.caption("Pydantic Schema Validation, Deterministic Action Calling & Live Tool Execution Loops")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Agent Controls")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Reset Session"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    uploaded_files = st.file_uploader("Upload Incident / SLA Docs (PDF)", type=["pdf"], accept_multiple_files=True)

    if st.button("Index Documents"):
        if uploaded_files:
            total_added = 0
            for file in uploaded_files:
                files = {"file": (file.name, file.getvalue(), "application/pdf")}
                try:
                    res = requests.post(f"{API_URL}/upload", files=files)
                    if res.status_code == 200:
                        data = res.json()
                        total_added += data.get("chunks_indexed", 0)
                        if file.name not in st.session_state.uploaded_files_list:
                            st.session_state.uploaded_files_list.append(file.name)
                    else:
                        st.error(f"Error on {file.name}: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
            st.success(f"Indexed successfully! Added {total_added} chunks.")
        else:
            st.warning("Upload at least one PDF.")

    st.divider()
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask an actionable query (e.g. 'Audit failed background jobs and dispatch high severity alert')...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_query,
        "target_doc": selected_doc
    }

    with st.chat_message("assistant"):
        with st.spinner("Extracting schema, evaluating tool dispatch triggers & executing actions..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    ans = data.get("final_conversational_answer", "")
                    structured = data.get("structured_data", {})
                    tools_run = data.get("executed_tools", [])

                    st.markdown(ans)

                    # Structured Schema Breakdown
                    st.markdown("### 📋 Enforced Pydantic Schema Output")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.info(f"**Summary:** {structured.get('summary')}")
                        st.write("**Identified Observations:**")
                        for iss in structured.get("identified_issues", []):
                            st.write(f"- {iss}")
                    with c2:
                        act_req = structured.get("action_required", False)
                        st.metric("Action Required", "🚨 YES" if act_req else "✅ NO")
                        if structured.get("recommended_action"):
                            rec = structured.get("recommended_action")
                            st.write(f"**Action Type:** `{rec.get('action_type')}`")
                            st.write(f"**Target:** `{rec.get('target_entity')}` | **Severity:** `{rec.get('severity')}`")

                    # Live Tool Execution Results
                    if tools_run:
                        st.markdown("### ⚡ Live Tool Execution Output")
                        for t in tools_run:
                            with st.expander(f"🛠️ Tool: `{t['tool_name']}` ({t['execution_status']})"):
                                st.json(t["output_data"])

                    st.session_state.messages.append({"role": "assistant", "content": ans})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
