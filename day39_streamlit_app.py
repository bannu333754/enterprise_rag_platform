import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 39 Plan-and-Solve Agentic RAG", layout="wide")

st.title("🎯 Day 39: Agentic Plan-and-Solve RAG")
st.caption("Dynamic Sub-Goal Decomposition, Step-by-Step Targeted Retrieval & Self-Reflective Verification")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Agent Planning Config")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    uploaded_files = st.file_uploader("Upload Documents (PDF)", type=["pdf"], accept_multiple_files=True)

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
    max_steps_val = st.slider("Max Planning Sub-Goals", min_value=2, max_value=5, value=3)
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Enter a complex multi-part query requiring planning and sequential reasoning...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_query,
        "target_doc": selected_doc,
        "max_steps": max_steps_val
    }

    with st.chat_message("assistant"):
        with st.spinner("Decomposing sub-goals, executing targeted retrievals & auditing verification..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=90)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("final_answer", "")
                    plan = data.get("execution_plan", [])
                    traces = data.get("step_traces", [])
                    verified = data.get("verification_passed", False)
                    rationale = data.get("verification_rationale", "")

                    st.markdown(answer)

                    # Agent Verification Status
                    badge = "🟢 Grounded Verification Passed" if verified else "🟡 Verification Warning"
                    st.caption(f"**Audit Status:** `{badge}` | {rationale}")

                    # Sequential Execution Breakdown
                    with st.expander(f"🧩 Sub-Goal Execution Traces ({len(traces)} Steps)"):
                        st.markdown("**Sub-Goal Plan:**")
                        for idx, step_name in enumerate(plan):
                            st.write(f"{idx+1}. `{step_name}`")
                        st.divider()

                        for trace in traces:
                            st.markdown(f"**Step {trace['step_number']}:** `{trace['sub_goal_query']}`")
                            st.info(f"**Intermediate Finding:** {trace['intermediate_finding']}")
                            st.caption(f"**Retrieved Evidence Chunk:** {trace['retrieved_chunk_summary']}")
                            st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
