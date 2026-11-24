import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 35 Multi-Modal Vision RAG", layout="wide")

st.title("👁️ Day 35: Multi-Modal RAG with Vision-Text Dual Streams")
st.caption("Synchronous Indexing of Text Passages and Visual Table/Diagram Layout Descriptors")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Multi-Modal Controls")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    uploaded_files = st.file_uploader("Upload Multi-Modal PDFs", type=["pdf"], accept_multiple_files=True)

    if st.button("Extract & Index Dual Streams"):
        if uploaded_files:
            total_added = 0
            with st.spinner("Extracting text passages and synthesizing visual table/diagram descriptors..."):
                for file in uploaded_files:
                    files = {"file": (file.name, file.getvalue(), "application/pdf")}
                    try:
                        res = requests.post(f"{API_URL}/upload", files=files, timeout=90)
                        if res.status_code == 200:
                            data = res.json()
                            total_added += data.get("dual_stream_chunks_indexed", 0)
                            if file.name not in st.session_state.uploaded_files_list:
                                st.session_state.uploaded_files_list.append(file.name)
                        else:
                            st.error(f"Error on {file.name}: {res.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
            st.success(f"Indexed {total_added} total multi-modal chunks!")
        else:
            st.warning("Upload at least one PDF.")

    st.divider()
    include_visuals = st.checkbox("Include Visual/Table Descriptors in Search", value=True)
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question about diagrams, layout tables, or textual details...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_query,
        "target_doc": selected_doc,
        "include_visual_nodes": include_visuals
    }

    with st.chat_message("assistant"):
        with st.spinner("Retrieving across text passages and visual layout representations..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("final_answer", "")
                    items = data.get("retrieved_items", [])

                    st.markdown(answer)

                    if items:
                        with st.expander(f"🧩 Multi-Modal Retrieved Evidence ({len(items)} Items)"):
                            for i, it in enumerate(items):
                                is_visual = it["modality"] == "VISUAL_TABLE_DIAGRAM"
                                badge = "🖼️ Visual / Table Descriptor" if is_visual else "📄 Text Passage"
                                st.markdown(f"**Item {i+1}** | `{badge}` | `{it['source']}` (p.{it['page']}) | Re-rank Score: `{it['score']}`")
                                if is_visual:
                                    st.info(it["content"])
                                else:
                                    st.write(it["content"])
                                st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
