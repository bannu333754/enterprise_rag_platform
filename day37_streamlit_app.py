import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 37 Contextual Compression RAG", layout="wide")

st.title("✂️ Day 37: Contextual Compression & Sentence-Level Token Pruning")
st.caption("Cross-Encoder Sentence Extraction, Redundancy Elimination & Dense Context Repacking")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Compression Settings")
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
    relevance_cutoff = st.slider("Sentence Relevance Threshold", min_value=-2.0, max_value=3.0, value=0.1, step=0.1)
    max_sentences = st.slider("Max Extracted Sentences", min_value=2, max_value=12, value=6)
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question to see sentence-level pruning and context repacking...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_query,
        "target_doc": selected_doc,
        "relevance_threshold": relevance_cutoff,
        "max_repacked_sentences": max_sentences
    }

    with st.chat_message("assistant"):
        with st.spinner("Extracting candidate passages, scoring sentences & pruning token noise..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("final_answer", "")
                    telemetry = data.get("telemetry", {})
                    extracted = data.get("extracted_sentences", [])

                    st.markdown(answer)

                    # Telemetry Dashboard
                    st.markdown("### 📊 Compression & Token Efficiency Telemetry")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Context Pruning", f"{telemetry.get('compression_ratio_pct', 0)}%", "Reduction")
                    with c2:
                        st.metric("Raw Characters", telemetry.get("raw_character_count", 0))
                    with c3:
                        st.metric("Repacked Characters", telemetry.get("compressed_character_count", 0))
                    with c4:
                        st.metric("Sentences Kept/Pruned", f"{telemetry.get('kept_sentences_count', 0)} / {telemetry.get('pruned_sentences_count', 0)}")

                    if extracted:
                        with st.expander(f"🧩 Repacked High-Density Sentences ({len(extracted)} preserved)"):
                            for i, s in enumerate(extracted):
                                st.markdown(f"**Sentence {i+1}** | `{s['source']}` (p.{s['page']}) | Score: `{s['score']}`")
                                st.write(f"\"{s['sentence']}\"")
                                st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
