import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 50 Self-Evolving RAG", layout="wide")

st.title("🌟 Day 50: Autonomous Self-Evolution & Active Learning RAG")
st.caption("Golden Dataset Synthesis, Hard-Negative Mining & Closed-Loop Hyperparameter Optimization")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Self-Evolution Hub")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    uploaded_files = st.file_uploader("Upload Knowledge Documents (PDF)", type=["pdf"], accept_multiple_files=True)

    if st.button("Index & Synthesize Golden Triples"):
        if uploaded_files:
            total_added = 0
            with st.spinner("Indexing and generating synthetic golden QA evaluation pairs..."):
                for file in uploaded_files:
                    files = {"file": (file.name, file.getvalue(), "application/pdf")}
                    try:
                        res = requests.post(f"{API_URL}/upload", files=files, timeout=90)
                        if res.status_code == 200:
                            data = res.json()
                            total_added += data.get("chunks_indexed", 0)
                            if file.name not in st.session_state.uploaded_files_list:
                                st.session_state.uploaded_files_list.append(file.name)
                        else:
                            st.error(f"Error on {file.name}: {res.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
            st.success(f"Indexed {total_added} chunks and populated Golden Evaluation Dataset!")
        else:
            st.warning("Upload at least one PDF.")

    st.divider()
    st.header("🔬 Autonomous Self-Tuner")
    if st.button("⚡ Run Hyperparameter Optimization"):
        with st.spinner("Evaluating grid configurations across Golden Benchmark..."):
            try:
                res = requests.post(f"{API_URL}/self-tune", timeout=60)
                if res.status_code == 200:
                    metrics = res.json()
                    st.success(f"Optimized to `{metrics.get('applied_version')}`!")
                    st.metric("Golden Samples Evaluated", metrics.get("golden_eval_samples"))
                    st.metric("Optimal Top-K", metrics.get("optimal_top_k"))
                    st.metric("Re-rank Threshold", metrics.get("optimal_rerank_threshold"))
                    st.metric("Simulated MRR", metrics.get("simulated_mrr_score"))
                else:
                    st.error(f"Tuning failed: {res.text}")
            except Exception as e:
                st.error(f"Request error: {e}")

    st.divider()
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question to test autonomous self-evolving retrieval and active feedback...")

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
        with st.spinner("Executing self-tuned retrieval pipeline & mining hard negatives..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    ans = data.get("final_answer", "")
                    q_id = data.get("query_id", "")
                    cfg = data.get("active_config", {})
                    chunk = data.get("retrieved_chunk", "")

                    st.markdown(ans)

                    # Dynamic Configuration Status Card
                    st.markdown("### 🧬 Active Hyperparameter Configuration")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Active Version", cfg.get("active_version", "v1.0.0"))
                    with c2:
                        st.metric("Tuned Top-K", cfg.get("top_k_retrieval", 4))
                    with c3:
                        st.metric("Re-rank Threshold", cfg.get("rerank_threshold", 0.25))

                    with st.expander("📄 Retrieved Context Evidence"):
                        st.caption(chunk)

                    # Active Learning Feedback Widget
                    col_fb1, col_fb2 = st.columns([1, 4])
                    with col_fb1:
                        if st.button("👍 Grounded", key=f"pos_{q_id}"):
                            requests.post(f"{API_URL}/feedback", json={"query_id": q_id, "is_accurate": True})
                            st.toast("Feedback recorded: Verified Grounded!")
                    with col_fb2:
                        if st.button("👎 Hallucinated / Misaligned", key=f"neg_{q_id}"):
                            requests.post(f"{API_URL}/feedback", json={"query_id": q_id, "is_accurate": False})
                            st.toast("Negative hard-mined into training pool!")

                    st.session_state.messages.append({"role": "assistant", "content": ans})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
