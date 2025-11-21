import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (if .env exists)
load_dotenv()

# ==========================================
# DIRECTORY CONFIGURATION
# ==========================================
# Uses pathlib for robust, cross-platform path handling
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"

# Input paths
RAW_DATA_DIR = DATA_DIR / "raw"
PDF_PATH = RAW_DATA_DIR / "qatar_test_doc.pdf"

# Output/Processed paths
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHUNKS_PATH = PROCESSED_DATA_DIR / "extracted_chunks.json"

IMAGES_DIR = DATA_DIR / "images"

# Vector Database
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
VECTOR_STORE_PATH = VECTOR_STORE_DIR / "faiss_index"

# Ensure all directories exist immediately upon import
directories = [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    VECTOR_STORE_DIR,
    IMAGES_DIR
]

for directory in directories:
    directory.mkdir(parents=True, exist_ok=True)


# ==========================================
# MODEL CONFIGURATION (LOCAL / LOW STORAGE)
# ==========================================

# 1. LLM Settings (Ollama)
# Switched to llama3.2:1b to fit within 11.2GB storage limit.
# It is much lighter than flan-t5-base for chat tasks.
LLM_MODEL_NAME = "llama3.2:1b"
OLLAMA_BASE_URL = "http://localhost:11434"

# 2. Embedding Settings (HuggingFace)
# 'all-MiniLM-L6-v2' is standard for efficiency (~80MB).
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"  # Use 'cuda' only if you have an NVIDIA GPU configured

# 3. Retrieval Settings
RETRIEVAL_K = 4           # Number of chunks to retrieve
CHUNK_SIZE = 1000         # Characters per chunk
CHUNK_OVERLAP = 200       # Overlap to preserve context

if __name__ == "__main__":
    print(f"Configuration loaded.")
    print(f"Project Root: {BASE_DIR}")
    print(f"PDF Target:   {PDF_PATH}")
    print(f"LLM Target:   {LLM_MODEL_NAME} (via {OLLAMA_BASE_URL})")