from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from rag.vectorstore import VectorStoreManager

class HybridRetriever:
    """
    Combines dense space semantic retrieval (FAISS) and sparse keyword retrieval (BM25)
    using LangChain's EnsembleRetriever.
    """
    
    def __init__(self, vectorstore_manager: VectorStoreManager):
        self.vectorstore_manager = vectorstore_manager
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None
        self.all_documents: List[Document] = []
        
    def _initialize_bm25(self, documents: List[Document]):
        """
        Initializes the BM25 keyword retriever.
        """
        if not documents:
            return
            
        print(f"Initializing BM25 retriever with {len(documents)} document chunks...")
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 4 # Retrieve top 4 using BM25
        
    def build_ensemble_retriever(self, documents: List[Document]):
        """
        Builds the combined EnsembleRetriever from the base FAISS and BM25 retrievers.
        This must be called whenever new documents are added to update the bm25 index.
        """
        if not documents:
            return
            
        # Keep track of all documents to re-build BM25 if needed
        self.all_documents.extend(documents)
        
        self._initialize_bm25(self.all_documents)
        
        # Get FAISS retriever
        faiss_retriever = self.vectorstore_manager.get_retriever(search_kwargs={"k": 4})
        
        # Combine FAISS and BM25 retrievers with equal weighting
        print("Building Ensemble Retriever (Weights: FAISS=0.5, BM25=0.5)...")
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
        
    def get_retriever(self) -> EnsembleRetriever:
        """
        Returns the active EnsembleRetriever.
        """
        if self.ensemble_retriever is None:
            raise ValueError("Ensemble Retriever not initialized. Please build it first by loading documents.")
        return self.ensemble_retriever
