from sentence_transformers import CrossEncoder
from typing import List
from langchain_core.documents import Document


class Reranker:
    """
    Cross-encoder re-ranker using ms-marco-MiniLM-L-6-v2.
    Takes the initial hybrid retrieval results and re-scores them
    using a cross-encoder for much higher precision.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading re-ranker model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        Re-ranks documents by relevance to the query using cross-encoder scoring.

        Args:
            query: The user's question.
            documents: List of retrieved documents to re-rank.
            top_k: Number of top documents to return after re-ranking.

        Returns:
            The top_k most relevant documents, sorted by cross-encoder score.
        """
        if not documents:
            return []

        # Create (query, document) pairs for scoring
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        # Sort by score descending and return top_k
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked[:top_k]]
