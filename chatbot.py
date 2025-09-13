import os
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
import docx
import pandas as pd
import re
from rank_bm25 import BM25Okapi

# ---------------- Gemini API ----------------
genai.configure(api_key="AIzaSyCA7UjyB8OlC0nZ1se-gorBIZv4JSyF-8Y")

MODEL_NAME = "models/gemma-3-12b-it"

# ---------------- Embedding Model ----------------
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- File Loader ----------------
def clean_text(text):
    # Remove multiple newlines, excessive spaces, headers/footers heuristics if any
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    return text

def load_file(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()
    text = ""

    try:
        if ext == "pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text())
            text = "\n".join(pages)
        elif ext == "docx":
            doc = docx.Document(uploaded_file)
            paras = [para.text for para in doc.paragraphs]
            text = "\n".join(paras)
        elif ext == "csv":
            df = pd.read_csv(uploaded_file)
            text = df.to_string()
        elif ext == "txt":
            text = uploaded_file.read().decode("utf-8")
        else:
            st.error(f"Unsupported file format: {uploaded_file.name}")
            return ""
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

    return clean_text(text)

# ---------------- Chunking ----------------
def semantic_chunking(text, file_name="", max_chunk_size=1000):
    paragraphs = text.split('\n\n')  # naive paragraph split
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip().replace('\n', ' ')
        if not para:
            continue
        if len(current_chunk) + len(para) > max_chunk_size:
            if current_chunk:
                chunks.append(f"[File: {file_name}] " + current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += " " + para

    if current_chunk:
        chunks.append(f"[File: {file_name}] " + current_chunk.strip())

    return chunks

# ---------------- Build FAISS Index ----------------
def build_faiss_index(chunks, embed_model):
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# ---------------- BM25 Retrieval ----------------
def build_bm25_index(chunks):
    tokenized_corpus = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

# ---------------- RAG Answer ----------------
def rag_answer_gemini(query, index, chunks, embed_model, bm25=None, k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    
    # Embedding-based FAISS search
    D, I = index.search(q_emb, k)
    faiss_hits = [chunks[i] for i in I[0]]
    
    # BM25 search (keyword-based)
    bm25_hits = []
    if bm25:
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_indices = bm25_scores.argsort()[-k:][::-1]
        bm25_hits = [chunks[i] for i in top_indices]
    
    # Combine hits (union & remove duplicates)
    combined_hits = list(dict.fromkeys(faiss_hits + bm25_hits))
    combined_hits_text = "\n".join(combined_hits[:k])  # Limit combined hits to k
    
    prompt = f"""
You are a helpful assistant. Use the context to answer the question concisely.

Context:
{combined_hits_text}

Question: {query}

Answer:
"""
    gen_model = genai.GenerativeModel(MODEL_NAME)
    response = gen_model.generate_content(prompt)
    return response.text

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Reader Chatbot", layout="wide")
st.title("📄 Reader Chatbot ")

uploaded_file = st.file_uploader(
    "Upload one file (PDF, DOCX, CSV, TXT)", 
    type=["pdf", "docx", "csv", "txt"]
)

if uploaded_file:
    if "kb_created" not in st.session_state:
        with st.spinner("Reading..."):
            text_data = load_file(uploaded_file)
            chunks = semantic_chunking(text_data, file_name=uploaded_file.name)
            index, embeddings = build_faiss_index(chunks, EMBED_MODEL)
            bm25 = build_bm25_index(chunks)
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.bm25 = bm25
            st.session_state.history = []
            st.session_state.kb_created = True
        st.success("✅ Completed, Ready to Answer you Questions !")

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    def submit_question():
        query = st.session_state.input_text
        if query:
            with st.spinner("Thinking..."):
                answer = rag_answer_gemini(
                    query,
                    st.session_state.index,
                    st.session_state.chunks,
                    EMBED_MODEL,
                    bm25=st.session_state.bm25,
                    k=3
                )
                st.session_state.history.insert(0, {"user": query, "bot": answer})
            st.session_state.input_text = ""

    st.text_input("What you want to know about the Document:", key="input_text", on_change=submit_question)

    for chat in st.session_state.history:
        st.markdown(f"**Bot:** {chat['bot']}")
        st.markdown(f"**You:** {chat['user']}\n")
