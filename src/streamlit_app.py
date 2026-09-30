import os
os.environ.setdefault("OPENAI_API_KEY", "dummy-not-used")

import json
import streamlit as st
import ollama
from pyserini.search.lucene import LuceneSearcher

# -----------------------------
# Configuration
# -----------------------------
THRESHOLD = 4.5
INDEX = "../target/indexes/bm25"
MODEL = "llama3.2:1b"

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Sydney Opera House Visitor Assistant",
    page_icon="🎭",
    layout="centered"
)

st.title("🎭 Sydney Opera House Visitor Assistant")
st.write("Ask me anything about visiting the Sydney Opera House.")

# -----------------------------
# Load BM25 searcher
# -----------------------------
@st.cache_resource
def load_searcher():
    return LuceneSearcher(INDEX)

searcher = load_searcher()

# -----------------------------
# Store chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# -----------------------------
# User input
# -----------------------------
question = st.chat_input("Ask a question...")

if question:

    # Show user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    # Retrieve relevant passages
    hits = searcher.search(question, 3)
    top_score = hits[0].score if hits else 0.0

    # -------------------------
    # Threshold check
    # -------------------------
    if not hits or top_score < THRESHOLD:

        answer = (
            "I'm sorry, I don't have enough information "
            "to answer that question."
        )

        with st.chat_message("assistant"):
            st.write(answer)

            with st.expander("Retrieval details"):
                st.write(f"BM25 score: {top_score:.2f}")
                st.write(f"Required threshold: {THRESHOLD}")

    else:

        # Get retrieved passage
        doc = json.loads(
            hits[0].lucene_document.get("raw")
        )

        passage = doc["contents"]

        # Same prompt as original chatbot
        prompt = f"""Answer the visitor's question using ONLY the information in the passage below. Be concise and friendly. If the passage doesn't fully answer the question, say what you do know from it.

Passage: {passage}

Question: {question}

Answer:"""

        # Generate answer using local Llama model
        with st.chat_message("assistant"):

            with st.spinner("Searching the Sydney Opera House information..."):

                response = ollama.generate(
                    model=MODEL,
                    prompt=prompt
                )

                answer = response["response"].strip()

            st.write(answer)

            # Useful for demonstrating RAG
            with st.expander("View retrieval details"):
                st.write(f"Source passage: {hits[0].docid}")
                st.write(f"BM25 retrieval score: {top_score:.2f}")
                st.write(f"Answer threshold: {THRESHOLD}")

                st.markdown("**Retrieved passage:**")
                st.write(passage)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })