import re
import pdfplumber
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from PIL import Image

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------
# Text Utilities
# -----------------------------
def deduplicate_text(text: str) -> str:
    """Remove repeated words or lines."""
    if not text:
        return text
    text = re.sub(r'([-_.])\1{2,}', r'\1', text)
    words = text.split()
    result, prev_word = [], None
    for w in words:
        if w != prev_word:
            result.append(w)
        prev_word = w
    joined = " ".join(result)
    joined = re.sub(r'\b(\w+)( \1\b)+', r'\1', joined)
    return joined.strip()

def safe_extract_text_from_page(page) -> str:
    """Safely extract text from a PDF page."""
    try:
        text = page.extract_text()
        if text and text.strip():
            return text
    except Exception:
        pass
    return ""

# -----------------------------
# PDF & Vectorstore
# -----------------------------
def build_vectorstore_from_uploaded(files) -> Dict[str, Any]:
    pdf_data = {}
    all_texts, all_metadatas = [], []

    for file in files:
        file.seek(0)
        name = file.name
        images, pages_text = [], []

        with pdfplumber.open(file) as pdf:
            for idx, page in enumerate(pdf.pages, 1):
                txt = safe_extract_text_from_page(page)
                pages_text.append({"page": idx, "text": txt})
                try:
                    img = page.to_image(resolution=150).original
                    images.append(img.convert("RGB"))
                except Exception:
                    pass

        pdf_data[name] = {"pages": pages_text, "images": images}

        # Collect text for vectorstore
        doc_text = "\n".join(p["text"] for p in pages_text if p["text"])
        if doc_text.strip():
            all_texts.append(doc_text)
            all_metadatas.append({"source": name})

    if not all_texts:
        return {"vectorstore": None, "pdf_data": pdf_data, "chunks_count": 0}

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts_for_index, metadatas_for_index = [], []
    for doc_text, meta in zip(all_texts, all_metadatas):
        chunks = splitter.split_text(doc_text)
        texts_for_index.extend(chunks)
        metadatas_for_index.extend([meta]*len(chunks))

    embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBED_MODEL)
    faiss_index = FAISS.from_texts(texts_for_index, embeddings, metadatas=metadatas_for_index)

    return {
        "vectorstore": faiss_index,
        "pdf_data": pdf_data,
        "chunks_count": len(texts_for_index)
    }

# -----------------------------
# Resize image for viewer
# -----------------------------
def resize_pil_image(pil_img, zoom: int):
    w, h = pil_img.size
    scale = zoom / 100
    resized = pil_img.resize((int(w*scale), int(h*scale)))
    return resized
