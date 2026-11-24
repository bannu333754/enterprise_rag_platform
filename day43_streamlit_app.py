import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 43 Speculative-Drafting RAG", layout="wide")

st.title("⚡ Day 43: Adaptive Speculative-Drafting RAG")
st.caption("Dual-Tier Architecture: Small-to-Large Speculative Decoding & Verification Engine")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Speculative Engine")
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
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question to see speculative drafting and target verification in action...")

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
        with st.spinner("Executing speculative draft decoding and target verification..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    ans = data.get("verified_answer", "")
                    draft = data.get("speculative_draft", "")
                    telemetry = data.get("telemetry", {})
                    chunk = data.get("retrieved_chunk", "")

                    st.markdown(ans)

                    # Performance Telemetry
                    st.markdown("### 🏎️ Speculative Execution Telemetry")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Draft Latency", f"{telemetry.get('draft_latency_ms', 0)} ms")
                    with c2:
                        st.metric("Verification Latency", f"{telemetry.get('verification_latency_ms', 0)} ms")
                    with c3:
                        st.metric("Total Latency", f"{telemetry.get('total_pipeline_ms', 0)} ms")
                    with c4:
                        status = telemetry.get("acceptance_status", "ACCEPTED")
                        badge = "🟢 Accepted" if status == "ACCEPTED" else "🔄 Refined"
                        st.metric("Draft Status", badge)

                    # Comparative Inspection
                    col_draft, col_context = st.columns(2)
                    with col_draft:
                        with st.expander("📝 Speculative Draft Output"):
                            st.write(draft)
                    with col_context:
                        with st.expander("📑 Retrieved Source Evidence"):
                            st.caption(chunk)

                    st.session_state.messages.append({"role": "assistant", "content": ans})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
