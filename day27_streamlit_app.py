import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 27 Late-Interaction ColBERT RAG", layout="wide")

st.title("🎯 Day 27: Late-Interaction & Token-Level MaxSim RAG")
st.caption("ColBERT-Style Token Embedding Matrices with Dynamic Query-Document Late Alignment")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ ColBERT Config")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    uploaded_files = st.file_uploader("Upload Documents (PDF)", type=["pdf"], accept_multiple_files=True)

    if st.button("Index Token Matrices"):
        if uploaded_files:
            total_added = 0
            with st.spinner("Tokenizing & computing per-token normalized embedding matrices..."):
                for file in uploaded_files:
                    files = {"file": (file.name, file.getvalue(), "application/pdf")}
                    try:
                        res = requests.post(f"{API_URL}/upload", files=files, timeout=90)
                        if res.status_code == 200:
                            data = res.json()
                            total_added += data.get("token_matrices_indexed", 0)
                            if file.name not in st.session_state.uploaded_files_list:
                                st.session_state.uploaded_files_list.append(file.name)
                        else:
                            st.error(f"Error on {file.name}: {res.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
            st.success(f"Indexed successfully! Stored {total_added} token matrices.")
        else:
            st.warning("Upload at least one PDF.")

    st.divider()
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question to see token-level MaxSim alignment and scoring...")

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
        with st.spinner("Computing token-to-token MaxSim dot products..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("final_answer", "")
                    items = data.get("retrieved_items", [])

                    st.markdown(answer)

                    if items:
                        with st.expander(f"🔬 Late-Interaction Alignment Details ({len(items)} Passages)"):
                            for i, it in enumerate(items):
                                st.markdown(f"**Rank {i+1}** | `{it['source']}` (p.{it['page']}) | **Cumulative MaxSim:** `{it['maxsim_score']}`")
                                st.write(f"\"{it['content']}\"")
                                
                                st.caption("**Top Token-to-Token Semantic Alignments:**")
                                align_text = " • ".join([f"`{a['query_token']}` ➔ `{a['matched_doc_token']}` ({a['similarity_score']})" for a in it['token_alignments']])
                                st.markdown(align_text)
                                st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
