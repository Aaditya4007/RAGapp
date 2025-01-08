import streamlit as st
from components.rag_engine import RAGEngine
from components.agentic_research import AgenticResearchAssistant
from models.groq_interface import GroqLLMInterface
import os

def main():
    # Page configuration
    st.set_page_config(
        page_title="Legal Research Platform",
        page_icon="⚖️",
        layout="wide"
    )

    # Sidebar for configuration
    st.sidebar.title("🏛️ Legal Research Platform")

    # API Key and Mode Selection
    groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    research_mode = st.sidebar.selectbox(
        "Research Mode",
        [
            "RAG Research",
            "Agentic Research",
            "Combined Research"
        ]
    )

    # Document Selection
    default_docs = [
        'C:/Users/User/Desktop/CODE FOLDER/genaiprj/docs/CA_AmendmentAct_2011_07092016_.pdf',
        'C:/Users/User/Desktop/CODE FOLDER/genaiprj/docs/Companies_Donation_to_National_Fund_Act_1951_.pdf',
        'C:/Users/User/Desktop/CODE FOLDER/genaiprj/docs/Cost_and_works_Accountants_Act_1959_.pdf',
        'C:/Users/User/Desktop/CODE FOLDER/genaiprj/docs/IBC_Version2_.pdf',
        'C:/Users/User/Desktop/CODE FOLDER/genaiprj/docs/LLP_Act_PDF_Version_2_.pdf',
        'C:/Users/User/Desktop/CODE FOLDER/genaiprj/docs/Partnership Act 1932_.pdf',
        'C:/Users/User/Desktop/CODE FOLDER/genaiprj/docs/Societies_Registration_Act_1860_.pdf',
        'C:/Users/User/Desktop/CODE FOLDER/genaiprj/docs/The competion Act_.pdf',
        'C:/Users/User/Desktop/CODE FOLDER/genaiprj/docs/The_Companies_Secretaries_Amendment_Act_2006_.pdf'
    ]

    selected_docs = st.sidebar.multiselect(
        "Select Documents",
        default_docs,
        default=default_docs
    )

    # Main content area
    st.title("🔍 Intelligent Legal Research Assistant")

    # Research query input
    query = st.text_input("Enter your legal research query")


    # Research button
    if st.button("Conduct Research"):
        if not groq_api_key:
            st.error("Please enter a Groq API Key")
            return

        if not selected_docs:
            st.error("Please select at least one document")
            return

        try:
            # Initialize components with API key
            os.environ['GROQ_API_KEY'] = groq_api_key

            # Perform research based on selected mode
            if research_mode == "RAG Research":
                rag_engine = RAGEngine().create_knowledge_base(selected_docs)
                context_docs = rag_engine.semantic_search(query)
                context = "\n\n".join([doc['document'] for doc in context_docs])
                response = rag_engine.generate_research_response(query, context)

                st.subheader("RAG Research Findings")
                st.write(response)

                st.subheader("Referenced Sources")
                for doc in context_docs:
                    with st.expander(f"Source from {doc['metadata'].get('source', 'Unknown')}"):
                        st.text(doc['document'][:1000])

            elif research_mode == "Agentic Research":
                research_assistant = AgenticResearchAssistant(selected_docs)
                research_result = research_assistant.multi_step_research(query)

                st.subheader("Agentic Research Report")
                st.write(research_result['final_report'])

                st.subheader("Sub-Query Insights")
                for result in research_result['research_results']:
                    with st.expander(result['sub_query']):
                        st.write(result['response'])

            elif research_mode == "Combined Research":
                rag_engine = RAGEngine().create_knowledge_base(selected_docs)
                context_docs = rag_engine.semantic_search(query)
                context = "\n\n".join([doc['document'] for doc in context_docs])
                rag_response = rag_engine.generate_research_response(query, context)

                research_assistant = AgenticResearchAssistant(selected_docs)
                agentic_result = research_assistant.multi_step_research(query)

                st.subheader("RAG Research Findings")
                st.write(rag_response)

                st.subheader("Agentic Research Report")
                st.write(agentic_result['final_report'])

                st.subheader("Referenced Sources")
                for doc in context_docs:
                    with st.expander(f"Source from {doc['metadata'].get('source', 'Unknown')}"):
                        st.text(doc['document'][:1000])

                st.subheader("Sub-Query Insights")
                for result in agentic_result['research_results']:
                    with st.expander(result['sub_query']):
                        st.write(result['response'])

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()