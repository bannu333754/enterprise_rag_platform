import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 44 Self-RAG Reflection Tokens", layout="wide")

st.title("🪞 Day 44: Self-RAG with Reflection Tokens & Critique Gates")
st.caption("Dynamic [Retrieve] Gating, [IsRel] Passage Filtering, [IsSup] Grounding & [IsUse] Scoring")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Self-RAG Configuration")
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

user_query = st.chat_input("Ask a question to see Self-RAG reflection tokens and critique gates in action...")

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
        with st.spinner("Evaluating [Retrieve] necessity, auditing [IsRel], and computing [IsSup]..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    ans = data.get("final_answer", "")
                    telemetry = data.get("telemetry", {})
                    critiques = data.get("passage_critiques", [])

                    st.markdown(ans)

                    # Reflection Tokens Dashboard
                    st.markdown("### 🏷️ Self-RAG Reflection & Critique Tokens")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        ret_tok = telemetry.get("retrieve_decision_token", "[Retrieve]")
                        color = "🟢" if ret_tok == "[Retrieve]" else "⚪"
                        st.metric("Retrieval Gate", f"{color} {ret_tok}")
                    with c2:
                        acc = telemetry.get("relevant_passages_accepted", 0)
                        total = telemetry.get("total_candidates_inspected", 0)
                        st.metric("[IsRel] Accepted", f"{acc} / {total} chunks")
                    with c3:
                        sup = telemetry.get("is_sup_token", "[Fully-Supported]")
                        st.metric("[IsSup] Grounding", sup)
                    with c4:
                        st.metric("[IsUse] Utility Score", f"{telemetry.get('is_use_score', 0):.1f} / 5.0")

                    if critiques:
                        with st.expander(f"🧐 Detailed Passage Critiques ({len(critiques)} Inspected)"):
                            for i, c in enumerate(critiques):
                                rel_badge = "✅ [Relevant]" if c["is_rel_token"] == "[Relevant]" else "❌ [Irrelevant]"
                                st.markdown(f"**Chunk {i+1}** | `{rel_badge}` | `{c['source']}` (p.{c['page']})")
                                st.caption(f"**Critic Rationale:** {c['relevance_rationale']}")
                                st.write(f"\"{c['text_snippet'][:180]}...\"")
                                st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": ans})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
