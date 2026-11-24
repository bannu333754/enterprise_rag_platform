import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 42 Self-Healing RAG", layout="wide")

st.title("🩺 Day 42: Self-Healing & Autonomous Query Correction RAG")
st.caption("Iterative Faithfulness Auditing, Hallucination Guardrails & Dynamic Query Reformulation")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Self-Healing Controls")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    uploaded_files = st.file_uploader("Upload Knowledge Documents (PDF)", type=["pdf"], accept_multiple_files=True)

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
    max_heal = st.slider("Max Self-Healing Loops", min_value=1, max_value=3, value=2)
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Enter a query to test hallucination detection and query correction...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_query,
        "target_doc": selected_doc,
        "max_heal_iterations": max_heal
    }

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, evaluating faithfulness & checking self-healing guardrails..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=90)
                if res.status_code == 200:
                    data = res.json()
                    ans = data.get("final_answer", "")
                    audit_trail = data.get("audit_trail", [])
                    healed = data.get("healing_triggered", False)
                    total_iter = data.get("total_iterations", 1)

                    st.markdown(ans)

                    # Healing Audit Status Banner
                    if healed:
                        st.warning(f"🩹 **Self-Healing Triggered:** Answer recovered after {total_iter} iterations.")
                    else:
                        st.success(f"✅ **Guardrail Passed:** Verified faithful on initial attempt.")

                    # Iteration Breakdown
                    with st.expander(f"🔍 Self-Healing & Faithfulness Audit Trail ({len(audit_trail)} Iterations)"):
                        for audit in audit_trail:
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.markdown(f"**Iteration {audit['iteration']}:** `{audit['executed_query']}`")
                            with col2:
                                st.metric("Faithfulness", f"{audit['faithfulness_score']:.2f}")
                            with col3:
                                st.write(f"Verdict: **{audit['audit_verdict']}**")

                            st.caption(f"**Retrieved Snippet:** {audit['retrieved_chunk']}")
                            st.write(f"**Draft Output:** {audit['candidate_answer']}")
                            if audit.get("reformulation_reason"):
                                st.info(f"🔄 {audit['reformulation_reason']}")
                            st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": ans})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
