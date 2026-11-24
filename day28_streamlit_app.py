import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 28 Self-RAG Reflection Tokens", layout="wide")

st.title("🪞 Day 28: Self-RAG with Reflection & Critique Tokens")
st.caption("Inline Reflection Tokens: [Retrieve] Thresholding, [IsRel] Filtering, [IsSup] Grounding & [IsUse] Utility")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Self-RAG Config")
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

user_query = st.chat_input("Test a query to inspect inline reflection tokens [Retrieve], [IsRel], [IsSup], [IsUse]...")

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
        with st.spinner("Self-RAG evaluating reflection tokens & critique verification..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("final_answer", "")
                    tokens = data.get("reflection_tokens", {})

                    st.markdown(answer)

                    # Reflection Tokens Dashboard
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        ret_dec = tokens.get("retrieve_decision", "[Retrieve: Yes]")
                        badge_color = "🟢" if "Yes" in ret_dec else "⚪"
                        st.metric("1. [Retrieve]", f"{badge_color} {ret_dec}")
                    with col2:
                        passages = tokens.get("passages_critique", [])
                        rel_count = sum(1 for p in passages if "[Relevant]" in p["is_rel"])
                        st.metric("2. [IsRel]", f"{rel_count}/{len(passages)} Passages")
                    with col3:
                        sup = tokens.get("is_sup", "[Fully Supported]")
                        st.metric("3. [IsSup]", f"🛡️ {sup}")
                    with col4:
                        util = tokens.get("is_use", 5)
                        st.metric("4. [IsUse] Utility", f"⭐ {util}/5")

                    with st.expander("🔍 Reflection & Critique Deep-Dive"):
                        st.markdown(f"**Retrieve Reason:** {tokens.get('retrieve_reason')}")
                        st.markdown(f"**Groundedness Rationale:** {tokens.get('groundedness_rationale')}")
                        st.divider()
                        st.markdown("**Per-Passage [IsRel] Critique:**")
                        for p in passages:
                            rel_badge = "✅" if p["is_rel"] == "[Relevant]" else "❌"
                            st.write(f"{rel_badge} `{p['is_rel']}` ({p['source']} p.{p['page']}) Score: `{p['rel_score']}` — \"{p['snippet']}\"")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
