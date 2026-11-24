import streamlit as st
import requests
import datetime
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 33 Temporal RAG", layout="wide")

st.title("⏳ Day 33: Temporal & Time-Decayed RAG")
st.caption("Exponential Time-Decay Scoring, Chronological Age Penalties & Date-Bound Filtering")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Temporal Controls")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    st.subheader("Document Ingestion")
    doc_date_input = st.date_input("Document Publication Date", value=datetime.date.today())
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if st.button("Index with Timestamp"):
        if uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            data = {"doc_date": doc_date_input.strftime("%Y-%m-%d")}
            try:
                res = requests.post(f"{API_URL}/upload", files=files, data=data)
                if res.status_code == 200:
                    data_json = res.json()
                    st.success(f"Indexed {data_json.get('chunks_indexed')} chunks dated {data_json.get('assigned_date')}!")
                    if uploaded_file.name not in st.session_state.uploaded_files_list:
                        st.session_state.uploaded_files_list.append(uploaded_file.name)
                else:
                    st.error(f"Upload failed: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
        else:
            st.warning("Select a PDF file.")

    st.divider()
    st.subheader("Decay Parameters")
    decay_rate = st.slider("Decay Rate (Lambda)", min_value=0.000, max_value=0.020, value=0.005, step=0.001, format="%.3f")
    enable_hard_cutoff = st.checkbox("Enable Hard Max Age Cutoff")
    max_days = None
    if enable_hard_cutoff:
        max_days = st.number_input("Max Allowed Age (Days)", min_value=1, max_value=3650, value=180)

    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a question to see time-decay scoring in action...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_query,
        "target_doc": selected_doc,
        "decay_lambda": decay_rate,
        "max_age_days": max_days
    }

    with st.chat_message("assistant"):
        with st.spinner("Calculating time-decay factors & chronological ranking..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("final_answer", "")
                    items = data.get("retrieved_items", [])

                    st.markdown(answer)

                    if items:
                        with st.expander(f"⏱️ Temporal Scoring Breakdown ({len(items)} Chunks)"):
                            for i, it in enumerate(items):
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Doc Date", f"{it['doc_date']}")
                                with col2:
                                    st.metric("Age", f"{it['age_days']} days")
                                with col3:
                                    st.metric("Decay Multiplier", f"{it['time_decay_factor']}")
                                with col4:
                                    st.metric("Final Score", f"{it['temporal_final_score']}")

                                st.caption(f"**Source:** `{it['source']}` | **Raw Cross-Encoder Score:** `{it['raw_rerank_score']}`")
                                st.write(f"\"{it['content']}\"")
                                st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
