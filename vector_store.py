import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import config

class VectorStoreManager:
    def __init__(self):
        self.index_path = config.VECTOR_STORE_PATH
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': config.EMBEDDING_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None

    def create_vector_store(self, chunks_data: Optional[List[Dict]] = None) -> None:
        """
        Creates a FAISS vector store from text chunks.
        If chunks_data is None, it loads from the processed JSON file.
        """
        # 1. Load data if not provided
        if not chunks_data:
            if not config.CHUNKS_PATH.exists():
                raise FileNotFoundError(f"No chunks found at {config.CHUNKS_PATH}. Run document_processor.py first.")
            
            print(f"Loading chunks from {config.CHUNKS_PATH}...")
            with open(config.CHUNKS_PATH, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)

        # 2. Convert to LangChain Documents
        # document_processor.py outputs 'page_content', ensuring compatibility here.
        documents = []
        for i, chunk in enumerate(chunks_data):
            doc = Document(
                page_content=chunk.get("page_content", ""),
                metadata=chunk.get("metadata", {})
            )
            # Add a unique ID to metadata for reference
            doc.metadata["chunk_id"] = i
            documents.append(doc)

        if not documents:
            print("No documents to index.")
            return

        # 3. Build Index
        print(f"Generating embeddings for {len(documents)} chunks using {config.EMBEDDING_MODEL_NAME}...")
        print(f"   (This runs locally on {config.EMBEDDING_DEVICE}, no API cost)")
        
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        # 4. Save to Disk
        self._save_to_disk()
        print(f"Vector store saved to {self.index_path}")

    def _save_to_disk(self):
        """Saves the FAISS index to the config path."""
        # FAISS save_local creates a folder, so we point to the parent directory 
        # and give it the folder name defined in config
        if self.vectorstore:
            self.vectorstore.save_local(str(self.index_path))

    def load_vector_store(self):
        """Loads the FAISS index from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found at {self.index_path}. Create it first.")
            
        print(f"Loading vector store from {self.index_path}...")
        self.vectorstore = FAISS.load_local(
            str(self.index_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self.vectorstore

    def get_retriever(self):
        """Returns a retriever for the RAG chain."""
        if not self.vectorstore:
            self.load_vector_store()
            
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.RETRIEVAL_K}
        )

if __name__ == "__main__":
    # Standalone execution to build the index
    try:
        manager = VectorStoreManager()
        manager.create_vector_store()
        
        # Test search
        print("\nTesting Retrieval:")
        retriever = manager.get_retriever()
        results = retriever.invoke("What are the fiscal projections for 2024?")
        
        for i, doc in enumerate(results):
            print(f"\nResult {i+1} (Page {doc.metadata.get('page')}):")
            print(doc.page_content[:150] + "...")
            
    except Exception as e:
        print(f"Error: {e}")