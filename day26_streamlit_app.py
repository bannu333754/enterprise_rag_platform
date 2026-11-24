import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 26 RAPTOR Tree RAG", layout="wide")

st.title("🌳 Day 26: RAPTOR (Recursive Tree-Organized Retrieval)")
st.caption("Hierarchical Leaf Facts + Macro Thematic Cluster Summaries in a Unified Search Tree")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Tree Indexing")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    uploaded_files = st.file_uploader("Upload Documents (PDF)", type=["pdf"], accept_multiple_files=True)

    if st.button("Build RAPTOR Tree"):
        if uploaded_files:
            total_added = 0
            with st.spinner("Extracting leaf chunks & generating Level 1 cluster abstractions..."):
                for file in uploaded_files:
                    files = {"file": (file.name, file.getvalue(), "application/pdf")}
                    try:
                        res = requests.post(f"{API_URL}/upload", files=files, timeout=90)
                        if res.status_code == 200:
                            data = res.json()
                            total_added += data.get("total_nodes_indexed", 0)
                            if file.name not in st.session_state.uploaded_files_list:
                                st.session_state.uploaded_files_list.append(file.name)
                        else:
                            st.error(f"Error on {file.name}: {res.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
            st.success(f"Tree built successfully! Indexed {total_added} total nodes.")
        else:
            st.warning("Upload at least one PDF.")

    st.divider()
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question (granular fact or broad thematic query)...")

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
        with st.spinner("Traversing RAPTOR tree layers & retrieving relevant nodes..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("final_answer", "")
                    nodes = data.get("retrieved_nodes", [])

                    st.markdown(answer)

                    if nodes:
                        with st.expander(f"🌲 RAPTOR Tree Traversal Evidence ({len(nodes)} nodes)"):
                            for i, node in enumerate(nodes):
                                badge = "🍃 Leaf Detail" if node["layer_type"] == "LEAF" else "🌳 Macro Thematic Summary"
                                st.markdown(f"**Node {i+1}** | **Type:** `{badge}` | **Re-rank Score:** `{node['score']}`")
                                st.caption(f"**Source:** `{node['source']}` (ID: `{node['node_id']}`)")
                                if node["layer_type"] == "TREE_SUMMARY":
                                    st.info(node["content"])
                                else:
                                    st.write(node["content"])
                                st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
