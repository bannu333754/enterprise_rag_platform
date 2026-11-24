import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 48 Contrastive HyDE RAG", layout="wide")

st.title("⚖️ Day 48: Multi-Hypothesis Contrastive RAG (Contrastive HyDE 2.0)")
st.caption("Dual-Hypothesis Synthetic Projection, Divergent Vector Space Probing & Multi-Perspective Synthesis")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

with st.sidebar:
    st.header("⚙️ Hypothesis Generation Config")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    uploaded_files = st.file_uploader("Upload Policy / Architecture PDFs", type=["pdf"], accept_multiple_files=True)

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
    contrast_mode = st.radio("Contrast Generation Mode", options=["POLARIZED", "COMPLEMENTARY"], index=0)
    doc_options = ["ALL"] + st.session_state.uploaded_files_list
    selected_doc = st.selectbox("Search Domain", options=doc_options)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Enter a query with trade-offs or alternative paradigms...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_query,
        "target_doc": selected_doc,
        "contrast_mode": contrast_mode
    }

    with st.chat_message("assistant"):
        with st.spinner("Generating contrastive hypotheses, probing vector space & synthesizing..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=90)
                if res.status_code == 200:
                    data = res.json()
                    verdict = data.get("synthesis_verdict", "")
                    hypos = data.get("hypotheses", [])
                    passages = data.get("retrieved_passages", [])

                    st.markdown(verdict)

                    # Multi-Hypothesis Inspection Cards
                    st.markdown("### 🧬 Synthetic Contrastive Hypotheses")
                    h_cols = st.columns(len(hypos)) if hypos else []
                    for idx, (col, h) in enumerate(zip(h_cols, hypos)):
                        with col:
                            st.info(f"**{h['label']}** (`{h['hypothesis_id']}`)")
                            st.write(f"\"{h['synthetic_passage']}\"")

                    # Cross-Ranked Passages Surfaced
                    with st.expander(f"📑 Surfaced Evidence via Contrastive Fusion ({len(passages)} Passages)"):
                        for p in passages:
                            st.markdown(f"**[{p['source']} p.{p['page']}]** | Matched via: `{p['matched_via']}` | Cross-Score: `{p['cross_score']}`")
                            st.write(f"\"{p['snippet']}\"")
                            st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": verdict})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
