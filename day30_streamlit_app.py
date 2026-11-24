import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 30 Enterprise Production RAG", layout="wide")

st.title("🚀 Day 30: Enterprise Production RAG & Telemetry Dashboard")
st.caption("End-to-End Latency Tracing, Automated RAG Triad Benchmarking & Health Telemetry")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Production Ops & Docs")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    # API Health Check
    try:
        health_res = requests.get(f"{API_URL}/", timeout=3)
        if health_res.status_code == 200:
            st.success("🟢 API Cluster: Online")
        else:
            st.warning("🟠 API Warning")
    except Exception:
        st.error("🔴 API Offline")

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
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Enter enterprise query to trace latency profiling and RAG Triad benchmarks...")

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
        with st.spinner("Processing request through production pipeline & auditing benchmarks..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("final_answer", "")
                    telemetry = data.get("telemetry", {})
                    benchmarks = data.get("benchmarks", {})
                    evidence = data.get("evidence_items", [])

                    st.markdown(answer)

                    # RAG Triad Benchmarks Panel
                    st.markdown("### 📊 RAG Triad Quality Benchmarks")
                    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                    with b_col1:
                        st.metric("Context Relevance", f"{int(benchmarks.get('context_relevance', 0) * 100)}%")
                    with b_col2:
                        st.metric("Groundedness", f"{int(benchmarks.get('groundedness', 0) * 100)}%")
                    with b_col3:
                        st.metric("Answer Relevance", f"{int(benchmarks.get('answer_relevance', 0) * 100)}%")
                    with b_col4:
                        st.metric("Triad Score", f"{benchmarks.get('triad_score', 0)}")

                    # Latency Telemetry Breakdown
                    st.markdown("### ⏱️ Latency & Execution Profiling")
                    t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5)
                    with t_col1:
                        st.metric("Retrieval", f"{telemetry.get('retrieval_ms', 0)} ms")
                    with t_col2:
                        st.metric("Re-ranking", f"{telemetry.get('reranking_ms', 0)} ms")
                    with t_col3:
                        st.metric("LLM Gen", f"{telemetry.get('generation_ms', 0)} ms")
                    with t_col4:
                        st.metric("Triad Audit", f"{telemetry.get('evaluation_ms', 0)} ms")
                    with t_col5:
                        st.metric("Total Latency", f"{telemetry.get('total_latency_ms', 0)} ms")

                    with st.expander("🔍 Audited Evidence Chunks"):
                        st.caption(f"**Audit Notes:** {benchmarks.get('audit_notes')}")
                        st.divider()
                        for i, ev in enumerate(evidence):
                            st.markdown(f"**Evidence {i+1}** | `{ev['source']}` (p.{ev['page']}) | Re-rank Score: `{ev['score']}`")
                            st.write(f"\"{ev['text']}\"")
                            st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
