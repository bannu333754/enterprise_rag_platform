import streamlit as st
import requests
import uuid
from datetime import date

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 49 Temporal State-Tracking RAG", layout="wide")

st.title("⏳ Day 49: Multi-Modal Temporal-Decayed State-Tracking RAG")
st.caption("Exponential Half-Life Scoring, Chronological Supersession Auditing & Version Evolution")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("📅 Dated Ingestion")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    doc_effective_date = st.date_input("Document Effective Date", value=date(2026, 1, 15))
    uploaded_file = st.file_uploader("Upload Dated PDF Document", type=["pdf"])

    if st.button("Index with Timestamp"):
        if uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            data = {"document_date": str(doc_effective_date)}
            try:
                res = requests.post(f"{API_URL}/upload", files=files, data=data)
                if res.status_code == 200:
                    r_json = res.json()
                    st.success(f"Indexed {r_json.get('chunks_indexed')} chunks stamped `{doc_effective_date}`!")
                    if uploaded_file.name not in st.session_state.uploaded_files_list:
                        st.session_state.uploaded_files_list.append(uploaded_file.name)
                else:
                    st.error(f"Upload error: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
        else:
            st.warning("Select a PDF file.")

    st.divider()
    st.header("⚙️ Decay Parameters")
    half_life = st.slider("Half-Life (Days)", min_value=30.0, max_value=720.0, value=180.0, step=30.0)
    alpha_weight = st.slider("Semantic vs Temporal Weight (α)", min_value=0.1, max_value=0.9, value=0.65, step=0.05)
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a temporal or evolving query (e.g., 'What is the current policy limit and what was it previously?')...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_query,
        "target_doc": selected_doc,
        "half_life_days": half_life,
        "alpha_semantic_weight": alpha_weight
    }

    with st.chat_message("assistant"):
        with st.spinner("Computing exponential decay, auditing state supersessions & tracing timeline..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    ans = data.get("final_answer", "")
                    timeline_chunks = data.get("retrieved_timeline_chunks", [])
                    audit = data.get("temporal_audit", {})

                    st.markdown(ans)

                    # State Evolution Audit Card
                    st.markdown("### 🔄 Chronological State & Supersession Audit")
                    transition = audit.get("state_transition_detected", False)
                    if transition:
                        st.warning("⚠️ **State Transition Detected:** Newer records supersede an earlier baseline.")
                    else:
                        st.success("✅ **Stable Policy State:** No conflicting state transition detected.")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.info(f"**Current Active State:**\n\n{audit.get('active_policy_state')}")
                    with c2:
                        hist = audit.get("historical_superseded_state")
                        st.caption(f"**Historical Baseline:**\n\n{hist if hist else 'No superseded predecessor found.'}")

                    # Timeline Progression Table
                    st.markdown("### 📈 Chronological Evidence Progression")
                    for c in timeline_chunks:
                        with st.expander(f"📅 [{c['doc_date']}] {c['source']} (p.{c['page']}) | Composite Score: {c['composite_score']}"):
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.write(f"**Age:** `{c['age_days']} days`")
                            with col_b:
                                st.write(f"**Semantic Relevance:** `{c['raw_similarity']}`")
                            with col_c:
                                st.write(f"**Decay Multiplier:** `{c['temporal_decay_factor']}`")
                            st.write(f"\"{c['content']}\"")

                    st.session_state.messages.append({"role": "assistant", "content": ans})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
