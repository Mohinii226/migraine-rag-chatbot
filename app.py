"""A chat web page for your papers.

    streamlit run app.py
"""
import streamlit as st

from rag import config
from rag.llm import describe_llm
from rag.pipeline import answer_question, load_store

st.set_page_config(page_title="Migraine Research Chatbot", page_icon="🧠")
st.title("🧠 Migraine Research Chatbot")
st.caption("Retrieval-augmented generation (RAG) over research papers. "
           "Research prototype - not medical advice.")


@st.cache_resource(show_spinner="Loading the index and embedding model...")
def get_store():
    return load_store()


def show_sources(passages):
    with st.expander("Sources used"):
        for number, p in enumerate(passages, start=1):
            st.markdown(f"**[{number}] {p['source']}, page {p['page']}** "
                        f"(similarity {p['score']:.2f})")
            st.write(p["text"])


try:
    store = get_store()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.subheader("Status")
    st.write(f"Chunks indexed: **{len(store.chunks)}**")
    st.write(f"Search backend: **{store.backend}**")
    st.write(f"Search mode: **{config.SEARCH_MODE}**")
    st.write(f"Answer mode: **{describe_llm()}**")
    st.write(f"Passages per question: **{config.TOP_K}**")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("passages"):
            show_sources(message["passages"])

question = st.chat_input("Ask a question about the papers")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the papers..."):
            result = answer_question(store, question)
        st.markdown(result["answer"])
        show_sources(result["passages"])
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "passages": result["passages"]})
