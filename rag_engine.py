from typing import List, Dict
from .document_loader import AdvancedDocumentLoader
from models.groq_interface import GroqLLMInterface
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class RAGEngine:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", chroma_path: str = "chroma_data"):
        """
        Initialize RAG Engine with embedding and vector store
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        self.llm = GroqLLMInterface()
        self.chroma_path = chroma_path

    def create_knowledge_base(self, document_paths: List[str]):
        """
        Create vector store from documents
        """
        # Load and preprocess documents
        documents = AdvancedDocumentLoader.load_documents(document_paths)
        processed_docs = AdvancedDocumentLoader.preprocess_documents(documents)

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=processed_docs,
            embedding=self.embeddings
        )
        return self

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform semantic search across documents
        """
        if not self.vectorstore:
            raise ValueError("Knowledge base not initialized. Call create_knowledge_base first.")

        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        return [
            {
                "document": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]

    def generate_research_response(self, query: str, context: str) -> str:
        """
        Generate comprehensive research response
        """
        # Prepare system and user messages
        system_msg = """You are an advanced legal research assistant.
        Provide comprehensive, well-referenced answers drawing from multiple documents.
        Always cite specific sources and highlight key insights."""

        # Generate response using Groq LLM
        response = self.llm.generate_response(
            system_message=system_msg,
            user_message=f"Context: {context}\n\nQuery: {query}"
        )

        return response

    def ask_rag_agent(self, query: str, top_k: int = 5) -> str:
        """
        Complete RAG pipeline: Retrieve and Generate a response.
        """
        # Step 1: Retrieve top-k documents
        retrieved_docs = self.semantic_search(query, top_k=top_k)
        context = "\n\n".join([doc["document"] for doc in retrieved_docs])

        # Step 2: Generate response using retrieved context
        response = self.generate_research_response(query=query, context=context)

        return response
