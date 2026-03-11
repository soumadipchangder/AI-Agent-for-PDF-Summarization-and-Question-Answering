import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# --- RAG Pipeline Imports ---
from rag.loader import load_single_pdf
from rag.chunking import split_documents
from rag.embeddings import get_embedding_model
from rag.vectorstore import VectorStoreManager
from tools.retrieval_tool import HybridRetriever
from agents.pdf_agent import PDFAgent

load_dotenv()

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="AI PDF Agent",
    page_icon="📄",
    layout="wide"
)

# ============================================================
# Session State Initialization
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []          # Chat history for display
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # LangGraph message history
if "agent" not in st.session_state:
    st.session_state.agent = None
if "document_summary" not in st.session_state:
    st.session_state.document_summary = ""
if "pdfs_uploaded" not in st.session_state:
    st.session_state.pdfs_uploaded = False
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# ============================================================
# Cached Resource: Embedding Model (loaded only once per session)
# ============================================================
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    """Load HuggingFace sentence-transformer model. Cached so it's only loaded once."""
    return get_embedding_model()

# ============================================================
# Core Processing Function
# ============================================================
def process_uploaded_pdfs(uploaded_files):
    """
    Ingests uploaded PDFs, builds the FAISS + BM25 hybrid retriever,
    initializes the LangGraph PDF agent, and generates an automatic summary.
    All done directly in process — no HTTP calls.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.sidebar.error("GROQ_API_KEY not found. Please add it to your .env file or HF Spaces secrets.")
        return

    st.session_state.is_processing = True

    try:
        all_docs = []
        file_names = []

        # Load PDFs from a temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            progress = st.sidebar.progress(0, text="Saving and loading PDFs...")

            for i, file in enumerate(uploaded_files):
                filepath = os.path.join(tmpdir, file.name)
                with open(filepath, "wb") as f:
                    f.write(file.getbuffer())

                docs = load_single_pdf(filepath)
                all_docs.extend(docs)
                file_names.append(file.name)
                progress.progress((i + 1) / len(uploaded_files), text=f"Loaded {file.name}")

            if not all_docs:
                st.sidebar.error("No valid text could be extracted from the uploaded PDFs.")
                return

            # --- Chunking ---
            progress.progress(0.5, text="Splitting documents into chunks...")
            chunks = split_documents(all_docs)

            # --- Embedding & Vector Store ---
            progress.progress(0.65, text="Building vector database...")
            embeddings = load_embedding_model()
            # Use an in-memory FAISS store (no persist dir) so it resets on each upload
            vectorstore_manager = VectorStoreManager(embeddings, persist_directory=None)
            vectorstore_manager.add_documents(chunks)

            # --- Hybrid Retriever ---
            progress.progress(0.8, text="Building hybrid retriever (FAISS + BM25)...")
            hybrid_retriever = HybridRetriever(vectorstore_manager)
            hybrid_retriever.build_ensemble_retriever(chunks)

            # --- LangGraph Agent ---
            progress.progress(0.9, text="Initializing AI agent...")
            st.session_state.agent = PDFAgent(
                retriever_callable=hybrid_retriever.get_retriever(),
                groq_api_key=groq_api_key
            )
            # Reset conversation history for new docs
            st.session_state.chat_history = []
            st.session_state.messages = []

            # --- Auto-summary ---
            progress.progress(0.95, text="Generating document summary...")
            summary_result = st.session_state.agent.run(
                "Provide a concise high-level summary of the document(s). What are the main topics and key takeaways?",
                chat_history=[]
            )
            st.session_state.document_summary = summary_result.get("answer", "Summary unavailable.")
            st.session_state.pdfs_uploaded = True
            st.session_state.is_processing = False

            progress.progress(1.0, text="✅ Ready!")
            st.sidebar.success(f"✅ Processed {len(file_names)} file(s)!")

    except Exception as e:
        st.sidebar.error(f"Error during processing: {e}")
        st.session_state.is_processing = False


# ============================================================
# UI Layout
# ============================================================
st.title("📄 Context-Aware AI PDF Assistant")
st.markdown("Upload your PDFs, review the automatic summary, and ask questions!")

# --- Sidebar: Upload ---
with st.sidebar:
    st.header("📁 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button(
        "⚙️ Process Documents",
        disabled=st.session_state.is_processing or not uploaded_files
    ):
        process_uploaded_pdfs(uploaded_files)
        st.rerun()

# --- Main Area ---
if st.session_state.pdfs_uploaded:

    # Document Summary
    with st.expander("📝 Document Summary", expanded=True):
        st.write(st.session_state.document_summary)

    st.divider()
    st.subheader("💬 Chat with your Documents")

    # Display past messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                with st.expander("📚 Sources"):
                    for cite in message["citations"]:
                        st.markdown(f"- `{cite}`")

    # Chat Input
    if prompt := st.chat_input("Ask a question about your documents..."):

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.agent.run(
                        prompt,
                        chat_history=st.session_state.chat_history
                    )

                    answer = result.get("answer", "Sorry, I could not generate an answer.")
                    raw_citations = result.get("citations", [])
                    st.session_state.chat_history = result.get("chat_history", [])

                    # Format citations: "filename.pdf (Page N)"
                    citations = []
                    for c in raw_citations:
                        src = os.path.basename(c.get("source", "Unknown"))
                        page = c.get("page", -1)
                        if page != -1:
                            citations.append(f"{src} (Page {page})")
                        else:
                            citations.append(src)
                    citations = list(set(citations))

                    st.markdown(answer)
                    if citations:
                        with st.expander("📚 Sources"):
                            for cite in citations:
                                st.markdown(f"- `{cite}`")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations
                    })

                except Exception as e:
                    err_msg = f"❌ Error: {e}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})

else:
    # Welcome screen when no docs are uploaded
    st.info("👈 Upload PDF documents using the sidebar to get started.")

    st.markdown("""
    ### What this assistant can do:
    | Feature | Description |
    |---------|-------------|
    | 📤 Multi-PDF Upload | Process multiple documents at once |
    | 📝 Auto Summary | Instant overview on upload |
    | 🔍 Hybrid Search | FAISS semantic + BM25 keyword retrieval |
    | 🤖 Self-Reflection | Agent critiques & refines its answers |
    | 📌 Source Citations | Exact filename + page number per answer |
    | 💬 Conversation Memory | Supports natural follow-up questions |
    """)
