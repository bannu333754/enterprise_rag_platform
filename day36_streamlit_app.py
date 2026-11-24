import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 36 Graph-Augmented RAG", layout="wide")

st.title("🕸️ Day 36: Graph-Augmented Entity-Disambiguated RAG")
st.caption("Knowledge Graph Triplet Extraction, Subgraph Traversal & Dense Vector Fusion")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

# Fetch live stats
kg_nodes, kg_edges = 0, 0
try:
    s_res = requests.get(f"{API_URL}/", timeout=3)
    if s_res.status_code == 200:
        s_data = s_res.json()
        kg_nodes = s_data.get("total_graph_nodes", 0)
        kg_edges = s_data.get("total_graph_edges", 0)
except Exception:
    pass

with st.sidebar:
    st.header("⚙️ Knowledge Graph Ops")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")
    st.metric("Graph Nodes", kg_nodes)
    st.metric("Graph Relational Edges", kg_edges)

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    uploaded_files = st.file_uploader("Upload Documents (PDF)", type=["pdf"], accept_multiple_files=True)

    if st.button("Build KG & Vector Embeddings"):
        if uploaded_files:
            total_added = 0
            with st.spinner("Extracting factual entity triplets & generating vector chunks..."):
                for file in uploaded_files:
                    files = {"file": (file.name, file.getvalue(), "application/pdf")}
                    try:
                        res = requests.post(f"{API_URL}/upload", files=files, timeout=90)
                        if res.status_code == 200:
                            data = res.json()
                            total_added += data.get("vector_chunks", 0)
                            if file.name not in st.session_state.uploaded_files_list:
                                st.session_state.uploaded_files_list.append(file.name)
                        else:
                            st.error(f"Error on {file.name}: {res.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
            st.success(f"Graph constructed! Added {total_added} vector chunks and extracted entity relations.")
            st.rerun()
        else:
            st.warning("Upload at least one PDF.")

    st.divider()
    max_hops = st.slider("Graph Traversal Depth (Hops)", min_value=1, max_value=3, value=2)
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a multi-hop relational question (e.g., 'How is X related to Y?')...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_query,
        "target_doc": selected_doc,
        "max_hops": max_hops
    }

    with st.chat_message("assistant"):
        with st.spinner("Extracting entities, traversing graph subgraphs, and re-ranking vectors..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("final_answer", "")
                    entities = data.get("extracted_entities", [])
                    triplets = data.get("graph_subgraph_triplets", [])
                    vectors = data.get("vector_evidence", [])

                    st.markdown(answer)

                    # Observability Panels
                    st.caption(f"🎯 **Identified Anchor Entities:** `{', '.join(entities) if entities else 'None'}`")

                    col1, col2 = st.columns(2)
                    with col1:
                        with st.expander(f"🕸️ Traversed Subgraph Paths ({len(triplets)})"):
                            if triplets:
                                for t in triplets:
                                    st.markdown(f"**({t['source_entity']})** ➔ `{t['relation']}` ➔ **({t['target_entity']})**")
                                    st.caption(f"Source: `{t['document_source']}`")
                            else:
                                st.write("No matching graph paths found.")

                    with col2:
                        with st.expander(f"📄 Unstructured Vector Evidence ({len(vectors)})"):
                            if vectors:
                                for v in vectors:
                                    st.markdown(f"`{v['source']}` (p.{v['page']}) | Re-rank Score: `{v['score']}`")
                                    st.caption(f"\"{v['text'][:140]}...\"")
                            else:
                                st.write("No vector evidence matched.")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
