import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 45 Corrective Multi-Agent Swarm", layout="wide")

st.title("🛟 Day 45: Corrective Multi-Agent Swarm (CRAG 2.0)")
st.caption("Confidence Auditing, Autonomous Fallback Search & Knowledge Boundary Disambiguation")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ CRAG 2.0 Controls")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    uploaded_files = st.file_uploader("Upload Internal PDFs", type=["pdf"], accept_multiple_files=True)

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
    conf_threshold = st.slider("Confidence Fallback Cutoff", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question to test internal retrieval, boundary auditing, and fallback...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_query,
        "target_doc": selected_doc,
        "confidence_threshold": conf_threshold
    }

    with st.chat_message("assistant"):
        with st.spinner("Auditing retrieval confidence, disambiguating boundaries & evaluating fallbacks..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    ans = data.get("final_answer", "")
                    telemetry = data.get("telemetry", {})
                    internal_ev = data.get("internal_evidence", "")
                    external_ev = data.get("external_evidence", "")

                    st.markdown(ans)

                    # CRAG Swarm Status Banner
                    st.markdown("### 🛡️ Corrective Swarm Telemetry")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        verdict = telemetry.get("retrieval_verdict", "CORRECT")
                        badge = "🟢" if verdict == "CORRECT" else ("🟡" if verdict == "AMBIGUOUS" else "🔴")
                        st.metric("Internal Verdict", f"{badge} {verdict}")
                    with c2:
                        st.metric("Confidence Score", f"{telemetry.get('internal_confidence_score', 0):.2f}")
                    with c3:
                        fallback = telemetry.get("fallback_search_triggered", False)
                        st.metric("Fallback Activated", "🚨 Yes" if fallback else "✅ No")
                    with c4:
                        st.metric("Attribution Mode", telemetry.get("attribution_mode", "INTERNAL_ONLY"))

                    # Evidence Inspect Panels
                    col_int, col_ext = st.columns(2)
                    with col_int:
                        with st.expander("📄 Internal Local Evidence"):
                            st.caption(internal_ev)
                    with col_ext:
                        with st.expander("🌐 External Fallback Synthesis"):
                            if external_ev:
                                st.write(f"**Query:** `{telemetry.get('fallback_query')}`")
                                st.write(external_ev)
                            else:
                                st.write("External fallback not needed for this query.")

                    st.session_state.messages.append({"role": "assistant", "content": ans})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
