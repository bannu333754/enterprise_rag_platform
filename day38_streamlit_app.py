import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 38 RAPTOR Hierarchical RAG", layout="wide")

st.title("🌳 Day 38: RAPTOR Tree-Organized Hierarchical RAG")
st.caption("Recursive Abstractive Summarization, Multi-Tier Chunk Aggregation & Layer-Aware Retrieval")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Tree Architecture")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    uploaded_files = st.file_uploader("Upload Documents (PDF)", type=["pdf"], accept_multiple_files=True)

    if st.button("Construct RAPTOR Tree"):
        if uploaded_files:
            total_nodes = 0
            with st.spinner("Extracting leaf chunks & recursively constructing Tier-1 and Tier-2 summaries..."):
                for file in uploaded_files:
                    files = {"file": (file.name, file.getvalue(), "application/pdf")}
                    try:
                        res = requests.post(f"{API_URL}/upload", files=files, timeout=120)
                        if res.status_code == 200:
                            data = res.json()
                            total_nodes += data.get("total_nodes_indexed", 0)
                            if file.name not in st.session_state.uploaded_files_list:
                                st.session_state.uploaded_files_list.append(file.name)
                        else:
                            st.error(f"Error on {file.name}: {res.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
            st.success(f"RAPTOR tree constructed! Added {total_nodes} nodes across all tiers.")
        else:
            st.warning("Upload at least one PDF.")

    st.divider()
    tier_filter = st.selectbox(
        "Retrieval Tier Focus",
        options=["ALL", "TIER_0_LEAF", "TIER_1_CLUSTER", "TIER_2_MACRO"]
    )
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a macro thematic query or a specific factual question...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_query,
        "target_doc": selected_doc,
        "target_tier": tier_filter
    }

    with st.chat_message("assistant"):
        with st.spinner("Traversing RAPTOR tree layers & evaluating relevance scores..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("final_answer", "")
                    nodes = data.get("retrieved_nodes", [])

                    st.markdown(answer)

                    if nodes:
                        with st.expander(f"🌲 Traversed Tree Nodes ({len(nodes)} Hits)"):
                            for i, node in enumerate(nodes):
                                tier = node["tier"]
                                if tier == "TIER_2_MACRO":
                                    badge = "🌳 Tier-2 Macro Summary"
                                elif tier == "TIER_1_CLUSTER":
                                    badge = "🌿 Tier-1 Cluster Summary"
                                else:
                                    badge = "🍃 Tier-0 Leaf Chunk"

                                st.markdown(f"**Hit {i+1}** | `{badge}` | Score: `{node['score']}` | Node ID: `{node['node_id']}`")
                                st.write(f"\"{node['content']}\"")
                                st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
