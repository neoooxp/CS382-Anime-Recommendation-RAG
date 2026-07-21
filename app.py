"""
RAG-Based AI Search System — upgraded interface.

This file implements:
- SentenceTransformer embeddings search with FAISS database.
- ChatGPT-like conversational chat interface using session state.
- Dynamic .txt document uploading and chunk-indexing.
- Search term highlighting and response latency tracking.
"""

import time
import re
import streamlit as st

from rag.ingest import load_documents, build_chunk_records
from rag.embed_store import VectorStore
from rag.generate import generate_answer

DATA_FOLDER = "data/sample_docs"

# Configure layout
st.set_page_config(
    page_title="Anime RAG Recommendation System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and main description
st.title("🤖 Anime RAG Chatbot")
st.caption("Ask questions about your favorite anime, search plot points, or discover recommendations based on your custom corpus!")


@st.cache_resource(show_spinner="Initializing SentenceTransformers & Indexing Anime Docs...")
def load_store():
    docs = load_documents(DATA_FOLDER)
    chunks = build_chunk_records(docs)
    store = VectorStore()
    store.build(chunks)
    return store, docs, chunks


# Initialize store
store, docs, chunks = load_store()

# Initialize session state for session-specific vector store and messages
if "vector_store" not in st.session_state:
    st.session_state.vector_store = store
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()


# Helper to highlight matching query terms in source chunks
def highlight_terms(text: str, query: str) -> str:
    if not query:
        return text
    # Clean the query and split into words
    words = re.findall(r'\w+', query.lower())
    # Exclude common stopwords to prevent highlighting everything
    stopwords = {"what", "is", "a", "the", "in", "of", "and", "to", "for", "with", "on", "at", "by", "an", "about", "from", "there", "some", "any", "this", "that", "it", "are", "you"}
    words = [w for w in words if len(w) > 2 and w not in stopwords]
    
    if not words:
        return text
        
    pattern = r"(" + "|".join(re.escape(w) for w in words) + r")"
    try:
        highlighted = re.sub(pattern, r"<mark style='background-color: #ffd700; color: black; padding: 2px 4px; border-radius: 3px;'>\1</mark>", text, flags=re.IGNORECASE)
        return highlighted
    except Exception:
        return text


# --- SIDEBAR: Upload and Configuration ---
with st.sidebar:
    st.header("⚙️ Settings & Configuration")
    
    # Retrieval configuration
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
    
    # Mode selection
    mode = st.radio(
        "Answer generation mode", 
        ["llm", "extractive"], 
        index=0,
        help="LLM mode uses Gemini (requires GEMINI_API_KEY). Extractive returns matching source chunks directly with no API key."
    )
    
    st.divider()
    
    # Dynamic document upload
    st.header("📄 Upload New Corpus")
    uploaded_file = st.file_uploader("Add custom documents (.txt)", type=["txt"])
    if uploaded_file is not None:
        if uploaded_file.name not in st.session_state.uploaded_files:
            with st.spinner(f"Splitting and embedding {uploaded_file.name}..."):
                # Read content
                content = uploaded_file.read().decode("utf-8")
                
                # Check for standard anime tag chunking
                from rag.ingest import chunk_text
                anime_chunks_raw = chunk_text(content)
                
                if anime_chunks_raw:
                    docs_dict = [{"title": uploaded_file.name, "text": content}]
                    new_chunks = build_chunk_records(docs_dict)
                else:
                    # Fallback paragraph-based splitter
                    from rag.ingest import Chunk
                    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                    new_chunks = []
                    for i, p in enumerate(paragraphs):
                        new_chunks.append(
                            Chunk(
                                chunk_id=f"upload-{uploaded_file.name}-{i}",
                                doc_title=uploaded_file.name,
                                text=p,
                                metadata={"title": uploaded_file.name, "source": uploaded_file.name}
                            )
                        )
                
                if new_chunks:
                    st.session_state.vector_store.add_chunks(new_chunks)
                    st.session_state.uploaded_files.add(uploaded_file.name)
                    st.toast(f"Successfully indexed {len(new_chunks)} chunks from {uploaded_file.name}!", icon="✅")
        else:
            st.sidebar.info(f"'{uploaded_file.name}' is already indexed.")

    st.divider()

    # Reset button
    if st.button("🗑️ Reset Chat & Corpus", type="secondary"):
        st.session_state.messages = []
        # Re-fetch a fresh store from cache
        st.session_state.vector_store, _, _ = load_store()
        st.session_state.uploaded_files = set()
        st.toast("Chat history and database reset completed!", icon="🗑️")
        st.rerun()

    st.divider()
    
    # Metadata Display
    total_docs = len(docs) + len(st.session_state.uploaded_files)
    total_chunks = len(st.session_state.vector_store.chunks)
    st.caption(f"Indexed **{total_docs}** documents → **{total_chunks}** chunks")
    
    with st.expander("Explore Document Index"):
        st.write("**Base Corpus Files:**")
        for d in docs:
            st.write(f"- {d['title']}")
        if st.session_state.uploaded_files:
            st.write("**Uploaded Files:**")
            for uf in st.session_state.uploaded_files:
                st.write(f"- {uf} (Uploaded)")


# --- MAIN CHAT INTERFACE ---

# Display chat history (older messages grouped in a collapsable expander)
if len(st.session_state.messages) > 2:
    with st.expander("📚 Show Past Conversation History", expanded=False):
        for msg in st.session_state.messages[:-2]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("latency_retrieval") is not None:
                    lat_r = msg["latency_retrieval"] * 1000
                    lat_g = msg["latency_generation"]
                    st.caption(f"⏱️ Retrieval: {lat_r:.1f}ms | Generation: {lat_g:.2f}s")
                    if msg.get("retrieved_chunks"):
                        with st.expander("Show Retrieved Sources & Match Highlights", expanded=False):
                            for chunk, score in msg["retrieved_chunks"]:
                                st.write(f"**{chunk.doc_title}**  ·  Similarity Score: `{score:.3f}`")
                                highlighted = highlight_terms(chunk.text, msg.get("query", ""))
                                st.markdown(f"<div style='border-left: 3px solid #6c757d; padding-left: 10px; margin-bottom: 10px; font-style: italic;'>{highlighted}</div>", unsafe_allow_html=True)

# Display the latest messages (last user prompt and assistant response) normally
latest_msgs = st.session_state.messages[-2:] if len(st.session_state.messages) >= 2 else st.session_state.messages
for msg in latest_msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display latency metadata for assistant messages
        if msg["role"] == "assistant" and msg.get("latency_retrieval") is not None:
            lat_r = msg["latency_retrieval"] * 1000
            lat_g = msg["latency_generation"]
            st.caption(f"⏱️ Retrieval: {lat_r:.1f}ms | Generation: {lat_g:.2f}s")
            
            # Display source chunks if present
            if msg.get("retrieved_chunks"):
                with st.expander("Show Retrieved Sources & Match Highlights", expanded=True):
                    for chunk, score in msg["retrieved_chunks"]:
                        st.write(f"**{chunk.doc_title}**  ·  Similarity Score: `{score:.3f}`")
                        # Highlight words matching the query
                        highlighted = highlight_terms(chunk.text, msg.get("query", ""))
                        st.markdown(f"<div style='border-left: 3px solid #6c757d; padding-left: 10px; margin-bottom: 10px; font-style: italic;'>{highlighted}</div>", unsafe_allow_html=True)



# Query input box
if prompt := st.chat_input("Ask a question about the anime..."):
    # Append and show user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Response generation
    with st.chat_message("assistant"):
        # Step 1: Retrieval with SentenceTransformers and FAISS
        t0 = time.perf_counter()
        retrieved = st.session_state.vector_store.query(prompt, top_k=top_k)
        t1 = time.perf_counter()
        latency_retrieval = t1 - t0

        # Step 2: Generation (Extractive or LLM-based via Gemini API)
        t2 = time.perf_counter()
        answer = generate_answer(
            query=prompt,
            retrieved=retrieved,
            mode=mode,
            history=st.session_state.messages[:-1]
        )
        t3 = time.perf_counter()
        latency_generation = t3 - t2

        st.markdown(answer)

        # Show metadata immediately
        lat_r = latency_retrieval * 1000
        st.caption(f"⏱️ Retrieval: {lat_r:.1f}ms | Generation: {latency_generation:.2f}s")

        # Show sources
        if retrieved:
            with st.expander("Show Retrieved Sources & Match Highlights"):
                for chunk, score in retrieved:
                    st.write(f"**{chunk.doc_title}**  ·  Similarity Score: `{score:.3f}`")
                    highlighted = highlight_terms(chunk.text, prompt)
                    st.markdown(f"<div style='border-left: 3px solid #6c757d; padding-left: 10px; margin-bottom: 10px; font-style: italic;'>{highlighted}</div>", unsafe_allow_html=True)

        # Save assistant's answer and search metadata to session history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "latency_retrieval": latency_retrieval,
            "latency_generation": latency_generation,
            "retrieved_chunks": retrieved,
            "query": prompt
        })

