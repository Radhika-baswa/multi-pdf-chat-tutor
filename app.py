import streamlit as st
from transformers import pipeline
import transformers.utils.logging as hf_logging
from typing import List, Dict
from utils import build_vectorstore_from_uploaded, resize_pil_image, deduplicate_text

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="📚 MultiPDF Tutor Chat", layout="wide")
st.title("📚 MultiPDF Tutor Chat")

hf_logging.set_verbosity_error()

# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = {}

# -----------------------------
# Model Settings
# -----------------------------
TEXT2TEXT_OPTIONS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
]

st.sidebar.title("⚙️ Model Settings")
model_choice = st.sidebar.selectbox("Choose model:", TEXT2TEXT_OPTIONS, index=1)
k_retrieve = st.sidebar.slider("Top-k retrieved chunks", 1, 10, 5)
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

# -----------------------------
# PDF Upload
# -----------------------------
uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if uploads:
    if [f.name for f in uploads] != [f.name for f in st.session_state.uploaded_files]:
        st.session_state.uploaded_files = uploads
        result = build_vectorstore_from_uploaded(uploads)
        st.session_state.vectorstore = result["vectorstore"]
        st.session_state.pdf_data = result["pdf_data"]
        st.session_state.chunks_count = result["chunks_count"]

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([1, 1])

# --- PDF Viewer ---
with col1:
    st.header("📄 PDF Viewer")
    if not st.session_state.uploaded_files:
        st.info("Upload PDFs to view here.")
    else:
        filenames = [f.name for f in st.session_state.uploaded_files]
        sel_file = st.selectbox("Select PDF:", filenames)
        zoom = st.slider("Zoom (%)", 50, 200, 100, 10)
        pdf_meta = st.session_state.pdf_data.get(sel_file, {})
        images = pdf_meta.get("images", [])
        if images:
            page = st.number_input("Page", 1, len(images), 1)
            pil_img = images[page-1]
            st.image(resize_pil_image(pil_img, zoom))

# --- Chat Section ---
with col2:
    st.header("💬 Chat with PDFs")
    user_input = st.chat_input("Ask about the PDFs...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        if not st.session_state.vectorstore:
            answer = "No retrievable text found in PDFs."
        else:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k_retrieve})
            docs = retriever.get_relevant_documents(user_input)

            # --- Improved deduplication ---
            seen_texts = set()
            unique_docs = []
            for d in docs:
                cleaned = d.page_content.strip().replace("\n", " ")
                if cleaned not in seen_texts:
                    seen_texts.add(cleaned)
                    unique_docs.append(d)

            # Build context from unique docs
            ctx = "\n".join(f"[{i}] {d.page_content[:400].strip()}" for i, d in enumerate(unique_docs, 1))

            # --- Smarter Prompt ---
            prompt = f"""
You are a helpful tutor. Use the following context to answer the question clearly and concisely. 
Do not repeat content from different chunks.
If context is insufficient, answer generally.

Answer Requirements:
- Summarize key ideas briefly.
- Avoid repeating same facts or sentences.
- Cite chunk numbers like [1], [2] if relevant.

Context:
{ctx}

Question: {user_input}
Answer:
"""

            try:
                gen = pipeline("text2text-generation", model=model_choice)
                out = gen(prompt, max_length=512, do_sample=False)
                answer = deduplicate_text(out[0]["generated_text"])
            except Exception as e:
                answer = f"Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Display chat history
    for m in st.session_state.messages:
        avatar = "😎" if m["role"] == "user" else "🤖"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])
