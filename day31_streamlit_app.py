import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Day 31 Distributed Sharded RAG", layout="wide")

st.title("🌐 Day 31: Distributed Sharded Multi-Index RAG")
st.caption("Asynchronous Parallel Fan-Out, Isolated Vector Shards & Reciprocal Rank Fusion (RRF)")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

# Fetch cluster status
available_shards = ["tech_engineering", "business_policy", "operations_general"]
shard_counts = {}
try:
    c_res = requests.get(f"{API_URL}/", timeout=3)
    if c_res.status_code == 200:
        c_data = c_res.json()
        available_shards = c_data.get("available_shards", available_shards)
        shard_counts = c_data.get("shard_chunk_counts", {})
except Exception:
    pass

with st.sidebar:
    st.header("⚙️ Sharded Cluster Ops")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    st.markdown("### 🗄️ Shard Inventory")
    for sh in available_shards:
        st.write(f"- `{sh}`: **{shard_counts.get(sh, 0)}** chunks")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    st.subheader("Upload to Specific Shard")
    dest_shard = st.selectbox("Destination Shard", options=available_shards)
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if st.button("Deploy to Shard"):
        if uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            data = {"assigned_shard": dest_shard}
            try:
                res = requests.post(f"{API_URL}/upload", files=files, data=data)
                if res.status_code == 200:
                    r_json = res.json()
                    st.success(f"Indexed {r_json.get('chunks_indexed')} chunks into `{dest_shard}`!")
                    st.rerun()
                else:
                    st.error(f"Upload failed: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
        else:
            st.warning("Select a PDF file first.")

    st.divider()
    selected_shards = st.multiselect("Query Shards", options=["ALL"] + available_shards, default=["ALL"])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a cross-domain query to trigger async map-reduce shard retrieval...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_query,
        "target_shards": selected_shards
    }

    with st.chat_message("assistant"):
        with st.spinner("Dispatching async fan-out queries & calculating Reciprocal Rank Fusion..."):
            try:
                res = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("final_answer", "")
                    shards_queried = data.get("active_shards_queried", [])
                    merged = data.get("merged_results", [])

                    st.markdown(answer)

                    st.caption(f"🎯 **Queried Shards:** `{' | '.join(shards_queried)}`")

                    if merged:
                        with st.expander(f"🧩 Merged Multi-Shard Evidence ({len(merged)} hits via RRF)"):
                            for i, hit in enumerate(merged):
                                st.markdown(f"**Hit {i+1}** | Shard: `{hit['shard']}` | Doc: `{hit['source']}` (p.{hit['page']}) | RRF Score: `{hit['rrf_score']}`")
                                st.write(f"\"{hit['content']}\"")
                                st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
