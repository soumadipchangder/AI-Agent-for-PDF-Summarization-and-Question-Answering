import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from rag.loader import load_pdfs_from_directory, load_single_pdf
from rag.chunking import split_documents
from rag.embeddings import get_embedding_model
from rag.vectorstore import VectorStoreManager
from tools.retrieval_tool import HybridRetriever
from agents.pdf_agent import PDFAgent, AgentState

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize RAG Components globally
embeddings = get_embedding_model()
vectorstore_manager = VectorStoreManager(embeddings, persist_directory="faiss_index")
hybrid_retriever = HybridRetriever(vectorstore_manager)

# Initialize Agent globally
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("WARNING: GROQ_API_KEY not found in environment. Agent functionality will fail.")

agent = None

summary_cache = "No PDFs have been uploaded yet."
chat_history = []

def initialize_retrievers():
    """Builds the ensemble retriever if vectorstore exists"""
    vectorstore = vectorstore_manager.get_vectorstore()
    if vectorstore:
         # To simplify for the example, we fetch the first few to initialize BM25 correctly.
         # In a real scenario, we'd store the raw documents alongside FAISS or re-ingest.
         # For this structure, we assume users hit /upload first to populate.
         pass
         

@app.route('/upload', methods=['POST'])
def upload_file():
    global summary_cache, agent, hybrid_retriever
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        files = request.files.getlist('file')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        all_docs = []
        file_names = []
        
        print("[UPLOAD] Starting file processing...")
        for file in files:
            if file and file.filename.endswith('.pdf'):
                try:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    print(f"[UPLOAD] Saved file: {filepath}")
                    
                    docs = load_single_pdf(filepath)
                    all_docs.extend(docs)
                    file_names.append(filename)
                    print(f"[UPLOAD] Loaded {len(docs)} pages from {filename}")
                except Exception as e:
                    print(f"[UPLOAD] Error loading {file.filename}: {e}")
                    return jsonify({'error': f'Error loading {file.filename}: {str(e)}'}), 400
                
        if not all_docs:
            print("[UPLOAD] No valid PDFs processed")
            return jsonify({'error': 'No valid PDFs processed'}), 400
            
        # Process documents
        print(f"[UPLOAD] Processing {len(all_docs)} documents...")
        chunks = split_documents(all_docs)
        print(f"[UPLOAD] Split into {len(chunks)} chunks")
        
        print("[UPLOAD] Adding documents to vectorstore...")
        vectorstore_manager.add_documents(chunks)
        
        # Update retrievers
        print("[UPLOAD] Building ensemble retriever...")
        hybrid_retriever.build_ensemble_retriever(chunks)
        
        # Validate API key before initializing agent
        if not groq_api_key:
            print("[UPLOAD] ERROR: GROQ_API_KEY is not set")
            return jsonify({'error': 'Backend not properly configured. GROQ_API_KEY is missing.'}), 500
        
        # Initialize Agent
        print("[UPLOAD] Initializing PDF Agent...")
        agent = PDFAgent(
            retriever_callable=hybrid_retriever.get_retriever(),
            groq_api_key=groq_api_key
        )
        print("[UPLOAD] Agent initialized successfully")
        
        # Generate automatic summary
        summary_prompt = "Provide a high-level summary of the uploaded document(s). What are the main topics and key takeaways?"
        print("[UPLOAD] Generating summary...")
        try:
            result = agent.run(summary_prompt)
            summary_cache = result["answer"]
            print("[UPLOAD] Summary generated successfully")
        except Exception as e:
            print(f"[UPLOAD] Error generating summary: {e}")
            summary_cache = f"Error generating summary: {str(e)}"
            
        print(f"[UPLOAD] Successfully processed {len(file_names)} files")
        return jsonify({
            'message': f'Successfully uploaded and processed {len(file_names)} files.',
            'files': file_names,
            'summary': summary_cache
        }), 200
        
    except Exception as e:
        print(f"[UPLOAD] Unexpected error: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    global agent, chat_history
    
    try:
        if not agent:
            print("[ASK] Agent not initialized")
            return jsonify({'error': 'Agent not initialized. Please upload a PDF first.'}), 400
            
        data = request.json
        if not data or 'question' not in data:
            print("[ASK] No question provided")
            return jsonify({'error': 'No question provided'}), 400
            
        question = data['question']
        print(f"[ASK] Processing question: {question[:100]}...")
        
        # Run agent
        result = agent.run(question, chat_history)
        
        # Update history
        chat_history = result.get("chat_history", [])
        
        # Flatten citations for serialization
        citations = []
        for citation in result.get("citations", []):
            try:
                # citation is a dict with source and page
                # format source nicely
                source = citation.get('source', 'Unknown')
                source_name = os.path.basename(source) if isinstance(source, str) else str(source)
                page = citation.get('page', -1)
                if page != -1:
                    citations.append(f"{source_name} (Page {page})")
                else:
                    citations.append(source_name)
            except Exception as cite_err:
                print(f"[ASK] Error processing citation: {cite_err}")
                
        print(f"[ASK] Successfully processed question with {len(citations)} citations")
        return jsonify({
            'answer': result.get('answer', ''),
            'citations': list(set(citations)) # Unique citations
        }), 200
    except Exception as e:
        print(f"[ASK] Error processing question: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/summary', methods=['GET'])
def get_summary():
    return jsonify({'summary': summary_cache}), 200

if __name__ == '__main__':
    # When running locally
    app.run(debug=True, host='0.0.0.0', port=5001)
