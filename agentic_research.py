from typing import List, Dict, Any
from .rag_engine import RAGEngine
from models.groq_interface import GroqLLMInterface

from typing import List, Dict, Any
from models.groq_interface import GroqLLMInterface

class AgenticResearchAssistant:
    def __init__(self, document_paths: List[str]):
        """
        Initialize Agentic Research Assistant
        """
        self.rag_engine = RAGEngine().create_knowledge_base(document_paths)
        self.llm = GroqLLMInterface()

    def multi_step_research(self, initial_query: str) -> Dict[str, Any]:
        """
        Perform multi-step research with reasoning and refinement
        """
        # Step 1: Initial query decomposition
        decomposition_prompt = f"""Decompose the following research query into sub-queries
        that will help comprehensively address the main question:

        Initial Query: {initial_query}

        Provide 3-5 specific sub-queries that break down the research needs."""

        sub_queries = self.llm.generate_response(
            system_message="You are an expert legal research strategist.",
            user_message=decomposition_prompt
        ).split('\n')

        # Step 2: Research each sub-query
        research_results = []
        for sub_query in sub_queries:
            # Semantic search
            context_docs = self.rag_engine.semantic_search(sub_query)

            # Generate detailed response for sub-query
            context = "\n\n".join([doc['document'] for doc in context_docs])
            sub_response = self.rag_engine.generate_research_response(sub_query, context)

            research_results.append({
                "sub_query": sub_query,
                "response": sub_response,
                "sources": context_docs
            })

        # Step 3: Synthesize final comprehensive response
        synthesis_prompt = f"""Synthesize a comprehensive research report
        from the following sub-query research results for the initial query: {initial_query}

        Sub-query Results:
        {research_results}

        Provide a cohesive, well-structured research report."""

        final_report = self.llm.generate_response(
            system_message="You are an expert legal research synthesizer.",
            user_message=synthesis_prompt
        )

        return {
            "initial_query": initial_query,
            "sub_queries": sub_queries,
            "research_results": research_results,
            "final_report": final_report
        }