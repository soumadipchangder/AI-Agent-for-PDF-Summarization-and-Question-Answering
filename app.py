import streamlit as st
import requests
import time

# Configure Backend URL
BACKEND_URL = "http://localhost:5001"

st.set_page_config(
    page_title="AI PDF Agent",
    page_icon="📄",
    layout="wide"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "document_summary" not in st.session_state:
    st.session_state.document_summary = ""
if "pdfs_uploaded" not in st.session_state:
    st.session_state.pdfs_uploaded = False

def upload_pdfs(files):
    """Sends PDFs to the Flask Backend"""
    if not files:
        return False
        
    st.session_state.is_processing = True
    
    try:
        # Prepare files for multipart/form-data request
        files_data = [('file', (f.name, f, f.type)) for f in files]
        
        with st.spinner("Uploading and processing documents... This may take a moment."):
            response = requests.post(f"{BACKEND_URL}/upload", files=files_data)
            response.raise_for_status()
            
            data = response.json()
            st.session_state.document_summary = data.get('summary', 'Summary not available.')
            st.session_state.pdfs_uploaded = True
            
            st.sidebar.success(f"Processing complete! {len(files)} files ready.")
            
    except requests.exceptions.ConnectionError:
        st.sidebar.error("Could not connect to the backend server. Is it running?")
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Error processing documents: {e}")
    finally:
        st.session_state.is_processing = False
        
def ask_question(question):
    """Sends query to Backend"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/ask", 
            json={"question": question}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting answer: {e}")
        return None

# UI Layout
st.title("📄 Context-Aware AI PDF Assistant")
st.markdown("Upload your PDFs, review the automatic summary, and ask questions!")

# Sidebar - Document Upload
with st.sidebar:
    st.header("Document Ingestion")
    
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents", disabled=st.session_state.is_processing or not uploaded_files):
        upload_pdfs(uploaded_files)

# Main Area
if st.session_state.pdfs_uploaded:
    
    # Expandable Summary Section
    with st.expander("📝 Document Summary", expanded=True):
        st.write(st.session_state.document_summary)
        
    st.divider()
        
    # Chat Interface
    st.subheader("Chat with your Documents")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message and message["citations"]:
                with st.expander("📚 Sources"):
                    for citation in message["citations"]:
                        st.markdown(f"- `{citation}`")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Add to state tracking
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask_question(prompt)
                
                if result:
                    answer = result.get('answer', 'Sorry, I failed to generate an answer.')
                    citations = result.get('citations', [])
                    
                    st.markdown(answer)
                    
                    if citations:
                        with st.expander("📚 Sources"):
                             for citation in citations:
                                st.markdown(f"- `{citation}`")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "citations": citations
                    })

else:
    # Instructions State
    st.info("👈 Please upload some PDF documents using the sidebar to begin.")
    
    st.markdown("""
    ### Features:
    - **Multi-Document Support**: Upload multiple PDFs at once.
    - **Automatic Summarization**: Get an instant overview of the uploaded content.
    - **Context-Aware QA**: Ask questions and receive detailed answers.
    - **Source Citations**: Every answer includes references to the original file and page number.
    - **Self-Reflective Reasoning**: Powered by LangGraph and Groq, the AI critiques its own answers to ensure accuracy.
    """)
