import argparse
import sys
from pathlib import Path
import time


import config
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from llm_qa import QAEngine

def print_header(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

def check_environment():
    """Verifies that input files and directories exist."""
    if not config.PDF_PATH.exists():
        print(f"Error: Input PDF not found at {config.PDF_PATH}")
        print("   Please place your 'qatar_test_doc.pdf' in data/raw/")
        sys.exit(1)
    
    # Verify Ollama connection (lightweight check)
    try:
        import requests
        response = requests.get(config.OLLAMA_BASE_URL)
        if response.status_code != 200:
            print("Warning: Ollama does not seem to be responding.")
    except:
        print("Warning: Could not connect to Ollama. Ensure 'ollama serve' is running.")

def run_ingestion(force=False):
    """Step 1: Extract text and tables from PDF."""
    print_header("STEP 1: Document Ingestion (PDF -> JSON)")
    
    if config.CHUNKS_PATH.exists() and not force:
        print(f"ℹChunks found at {config.CHUNKS_PATH.name}")
        print("   Skipping ingestion to save time. Use --force to overwrite.")
        return

    try:
        start_time = time.time()
        processor = DocumentProcessor()
        chunks = processor.process_pdf()
        processor.save_chunks(chunks)
        print(f"Ingestion Complete in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Ingestion Failed: {e}")
        sys.exit(1)

def run_indexing(force=False):
    """Step 2: Create Embeddings and FAISS Index."""
    print_header("STEP 2: Vector Indexing (JSON -> FAISS)")
    
    if config.VECTOR_STORE_PATH.exists() and not force:
        print(f"ℹIndex found at {config.VECTOR_STORE_PATH.name}")
        print("   Skipping indexing to save time. Use --force to overwrite.")
        return

    try:
        start_time = time.time()
        manager = VectorStoreManager()
        # This loads the JSON we created in Step 1
        manager.create_vector_store()
        print(f"Indexing Complete in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Indexing Failed: {e}")
        sys.exit(1)

def run_inference(query):
    """Step 3: Run the RAG Chain."""
    print_header(f"STEP 3: QA Inference\n❓ Query: {query}")

    try:
        # 1. Load Retrieval Engine
        manager = VectorStoreManager()
        retriever = manager.get_retriever()
        
        # 2. Retrieve relevant docs
        print("Retrieving relevant context...")
        retrieved_docs = retriever.invoke(query)
        
        if not retrieved_docs:
            print("No relevant documents found for this query.")
            return

        # 3. Run LLM
        print(f"Generating answer with {config.LLM_MODEL_NAME}...")
        start_time = time.time()
        
        qa_engine = QAEngine()
        result = qa_engine.answer_question(query, retrieved_docs)
        
        duration = time.time() - start_time

        # 4. Display Results
        print_header("RESULT")
        print(f"Answer ({duration:.2f}s):\n{result['answer']}\n")
        
        print("-" * 40)
        print("Evidence Used:")
        for cit in result['citations']:
            source = cit.get('source', 'Unknown')
            page = cit.get('page', '?')
            print(f"   • [{source}, Page {page}]")
            
    except Exception as e:
        print(f"Inference Failed: {e}")
        print("   (Double check that Ollama is running)")

def main():
    parser = argparse.ArgumentParser(description="Run the IMF Document Intelligence Pipeline")
    
    # Arguments to control flow
    parser.add_argument("--force", action="store_true", help="Force re-ingestion and re-indexing")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip PDF processing (run QA only)")
    parser.add_argument("--query", type=str, default="What are the fiscal projections for 2024?", help="Question to ask")
    
    args = parser.parse_args()

    check_environment()

    # Pipeline Execution
    if not args.skip_ingest:
        run_ingestion(force=args.force)
        run_indexing(force=args.force)
    
    run_inference(args.query)

if __name__ == "__main__":
    main()