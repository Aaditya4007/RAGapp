import os
from typing import List, Dict, Any
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader
)
from langchain.docstore.document import Document

class AdvancedDocumentLoader:
    @staticmethod
    def get_loader_for_file(file_path: str):
        """
        Select appropriate loader based on file extension
        """
        ext = os.path.splitext(file_path)[1].lower()
        loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.html': UnstructuredHTMLLoader
        }
        return loaders.get(ext, TextLoader)

    @classmethod
    def load_documents(cls, file_paths: List[str], batch_size: int = 10) -> List[Document]:
        """
        Load multiple documents from given file paths in batches
        """
        all_documents = []

        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i:i+batch_size]
            batch_docs = []

            for file_path in batch_paths:
                try:
                    # Validate file exists
                    if not os.path.exists(file_path):
                        print(f"Warning: File not found - {file_path}")
                        continue

                    # Select and use appropriate loader
                    loader_class = cls.get_loader_for_file(file_path)
                    loader = loader_class(file_path)
                    documents = loader.load()

                    # Attach source metadata
                    for doc in documents:
                        doc.metadata['source'] = file_path

                    batch_docs.extend(documents)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

            # Preprocess documents in the batch
            processed_docs = cls.preprocess_documents(batch_docs)
            all_documents.extend(processed_docs)

        return all_documents

    @staticmethod
    def preprocess_documents(documents: List[Document], chunk_size: int = 5000, chunk_overlap: int = 200) -> List[Document]:
        """
        Preprocess documents with text splitting
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        return text_splitter.split_documents(documents)