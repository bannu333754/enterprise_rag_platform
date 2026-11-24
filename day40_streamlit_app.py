import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 40 Multi-Agent Swarm RAG", layout="wide")

st.title("🤖 Day 40: Multi-Agent Swarm Collaborative RAG")
st.caption("Role-Specialized Swarm Agents, Conflict Auditing & Executive Consensus Arbitration")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Swarm Coordination")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    uploaded_files = st.file_uploader("Upload Policy & Tech PDFs", type=["pdf"], accept_multiple_files=True)

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

user_query = st.chat_input("Ask a complex strategic query (e.g., 'What are the trade-offs of migrating to microservices?')...")

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
        with st.spinner("Dispatching role-specialized agents, evaluating evidence & resolving trade-offs..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=90)
                if res.status_code == 200:
                    data = res.json()
                    verdict = data.get("consensus_verdict", "")
                    evals = data.get("agent_evaluations", [])
                    conflicts = data.get("identified_conflicts", [])

                    st.markdown(verdict)

                    if conflicts:
                        st.markdown("### ⚖️ Identified Cross-Agent Trade-offs & Conflicts")
                        for c in conflicts:
                            st.warning(f"⚠️ {c}")

                    st.markdown("### 👥 Swarm Specialist Assessments")
                    cols = st.columns(len(evals)) if evals else []
                    for idx, (col, ev) in enumerate(zip(cols, evals)):
                        with col:
                            st.info(f"**{ev['role_name']}**")
                            st.write(ev["domain_assessment"])
                            with st.expander("Retrieved Evidence"):
                                st.caption(ev["retrieved_evidence"])

                    st.session_state.messages.append({"role": "assistant", "content": verdict})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
