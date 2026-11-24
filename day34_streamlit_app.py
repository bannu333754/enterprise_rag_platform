import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 34 Self-Querying Metadata RAG", layout="wide")

st.title("🔍 Day 34: Self-Querying & Metadata-Aware RAG")
st.caption("LLM-Driven Query Decomposition: Semantic Extraction + Dynamic ChromaDB AST Metadata Filtering")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Ingestion & Schema")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    st.subheader("Document Metadata Tagging")
    cat = st.selectbox("Category", ["Technical", "Business", "Policy", "Compliance"])
    dept = st.selectbox("Department", ["Engineering", "Finance", "Legal", "Operations"])
    year = st.number_input("Publication Year", min_value=2020, max_value=2027, value=2025)
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

    if st.button("Index with Metadata"):
        if uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            data = {
                "category": cat,
                "department": dept,
                "pub_year": year
            }
            try:
                res = requests.post(f"{API_URL}/upload", files=files, data=data)
                if res.status_code == 200:
                    r_json = res.json()
                    st.success(f"Indexed {r_json.get('chunks_indexed')} chunks with metadata tags!")
                    if uploaded_file.name not in st.session_state.uploaded_files_list:
                        st.session_state.uploaded_files_list.append(uploaded_file.name)
                else:
                    st.error(f"Upload failed: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
        else:
            st.warning("Please upload a PDF file.")

    st.divider()
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Try queries like: 'What are the 2025 engineering policy requirements for deployments?'")

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
        with st.spinner("Decomposing natural query into semantic intent & structured filters..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("final_answer", "")
                    plan = data.get("decomposition_plan", {})
                    items = data.get("retrieved_items", [])

                    st.markdown(answer)

                    # AST Filter Decomposition Panel
                    st.markdown("### 🧬 Query Decomposition & AST Filters")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Clean Semantic Query", f"\"{plan.get('semantic_query')}\"")
                    with c2:
                        st.metric("Target Dept", f"{plan.get('target_department') or 'Any'}")
                    with c3:
                        st.metric("Category", f"{plan.get('detected_category') or 'Any'}")
                    with c4:
                        st.metric("Year Filter", f"{plan.get('min_year') or 'Any'}")

                    st.code(f"ChromaDB Query Filter: {plan.get('raw_filter_json')}", language="json")

                    if items:
                        with st.expander(f"📑 Filter-Matched Evidence Chunks ({len(items)})"):
                            for i, it in enumerate(items):
                                st.markdown(f"**Doc {i+1}** | `{it['source']}` (p.{it['page']}) | Dept: `{it['department']}` | Cat: `{it['category']}` | Year: `{it['year']}` | Score: `{it['score']}`")
                                st.write(f"\"{it['content']}\"")
                                st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
