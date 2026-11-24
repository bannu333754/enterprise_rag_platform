import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 47 Multi-Index Adaptive Router RAG", layout="wide")

st.title("🎛️ Day 47: Adaptive Multi-Index Router & Cost-Optimized RAG")
st.caption("Complexity-Driven Index Dispatching, Latency Budget Enforcement & Cloud Compute Optimization")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Router & SLA Config")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    uploaded_files = st.file_uploader("Upload Documents (PDF)", type=["pdf"], accept_multiple_files=True)

    if st.button("Index Documents (Lexical + Vector)"):
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
            st.success(f"Dual-Indexed successfully! Added {total_added} chunks.")
        else:
            st.warning("Upload at least one PDF.")

    st.divider()
    latency_sla = st.slider("Target Max Latency SLA (ms)", min_value=200.0, max_value=2500.0, value=900.0, step=50.0)
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Test query (e.g. 'error code 404', 'define latency', or 'compare architectural trade-offs')...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_query,
        "target_doc": selected_doc,
        "max_latency_budget_ms": latency_sla
    }

    with st.chat_message("assistant"):
        with st.spinner("Classifying complexity, dispatching optimal tier & optimizing cost..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    ans = data.get("final_answer", "")
                    telemetry = data.get("telemetry", {})
                    evidence = data.get("retrieved_evidence_summary", "")

                    st.markdown(ans)

                    # Routing Telemetry Panel
                    st.markdown("### 🚦 Adaptive Routing & Cost Telemetry")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        cx = telemetry.get("detected_complexity", "SEMANTIC_FACT")
                        st.metric("Detected Complexity", cx)
                    with c2:
                        tier = telemetry.get("routed_tier", "TIER_2_DENSE_VECTOR")
                        if tier == "TIER_1_LEXICAL":
                            badge = "⚡ Tier 1: Lexical"
                        elif tier == "TIER_2_DENSE_VECTOR":
                            badge = "🔹 Tier 2: Vector"
                        else:
                            badge = "🔬 Tier 3: Deep Re-rank"
                        st.metric("Dispatched Tier", badge)
                    with c3:
                        tot_ms = telemetry.get("total_pipeline_ms", 0)
                        sla_ok = telemetry.get("sla_met", True)
                        st.metric("Total Latency", f"{tot_ms:.1f} ms", delta="SLA Met" if sla_ok else "SLA Breached", delta_color="normal" if sla_ok else "inverse")
                    with c4:
                        cost = telemetry.get("simulated_compute_cost_usd", 0.0)
                        st.metric("Simulated Cost", f"${cost:.5f}")

                    # Latency Waterfall Breakdown
                    with st.expander("⏱️ Latency Waterfall & Pipeline Stages"):
                        st.write(f"• **Routing Classification:** `{telemetry.get('routing_latency_ms', 0)} ms`")
                        st.write(f"• **Index Retrieval:** `{telemetry.get('retrieval_latency_ms', 0)} ms`")
                        st.write(f"• **LLM Generation:** `{telemetry.get('synthesis_latency_ms', 0)} ms`")
                        st.divider()
                        st.markdown("**Retrieved Passages:**")
                        st.caption(evidence)

                    st.session_state.messages.append({"role": "assistant", "content": ans})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
