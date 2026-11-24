import streamlit as st
import requests
import json
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 32 Streaming RAG", layout="wide")

st.title("⚡ Day 32: Streaming RAG with SSE & Progressive Citations")
st.caption("Low-Latency Token-by-Token Streaming with Real-Time Pre-Flight Citation Injection")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Streaming Config")
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
        if "citations" in msg and msg["citations"]:
            with st.expander("📚 Pinned Citations"):
                for i, c in enumerate(msg["citations"]):
                    st.markdown(f"**[{i+1}] {c['source']}** (Page {c['page']}) | Re-rank Score: `{c['score']}`")
                    st.caption(f"\"{c['text']}\"")

user_query = st.chat_input("Ask a question to see real-time streaming token generation...")

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
        text_placeholder = st.empty()
        citations_placeholder = st.empty()
        full_text = ""
        citations_data = []

        try:
            with requests.post(f"{API_URL}/chat/stream", json=payload, stream=True, timeout=60) as resp:
                if resp.status_code == 200:
                    for line in resp.iter_lines():
                        if line:
                            decoded = line.decode('utf-8')
                            if decoded.startswith("data: "):
                                data_str = decoded[6:]
                                data = json.loads(data_str)

                                event_type = data.get("event")
                                if event_type == "citations_metadata":
                                    citations_data = data.get("citations", [])
                                    if citations_data:
                                        with citations_placeholder.container():
                                            st.caption(f"⚡ Pre-flight Citation Lock: {len(citations_data)} passages identified")
                                elif event_type == "token":
                                    delta = data.get("delta", "")
                                    full_text += delta
                                    text_placeholder.markdown(full_text + "▌")
                                elif event_type == "done":
                                    text_placeholder.markdown(full_text)
                                elif event_type == "error":
                                    st.error(f"Stream error: {data.get('message')}")
                    
                    text_placeholder.markdown(full_text)

                    if citations_data:
                        with st.expander("📚 Pinned Document Citations"):
                            for i, c in enumerate(citations_data):
                                st.markdown(f"**[{i+1}] {c['source']}** (Page {c['page']}) | Re-rank Score: `{c['score']}`")
                                st.caption(f"\"{c['text']}\"")
                                st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_text,
                        "citations": citations_data
                    })
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")
        except Exception as e:
            st.error(f"Streaming failed: {str(e)}")
