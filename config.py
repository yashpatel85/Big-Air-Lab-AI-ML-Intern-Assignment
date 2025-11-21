import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 📁 DIRECTORY CONFIGURATION (ALWAYS RUNS)
# ==========================================
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
# CRITICAL: This variable must be defined globally, outside any if/else blocks
VECTOR_STORE_PATH = VECTOR_STORE_DIR / "faiss_index"

# Ensure directories exist
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
# 🧠 MODEL CONFIGURATION
# ==========================================

# Toggle this for Cloud Deployment via Streamlit Secrets or Env Vars
# Options: "local" (Ollama), "cloud" (Groq)
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "local")

# 1. LLM Settings
if DEPLOYMENT_MODE == "local":
    LLM_MODEL_NAME = "llama3.2:1b"
    OLLAMA_BASE_URL = "http://localhost:11434"
    GROQ_API_KEY = None
else:
    # Cloud Mode (Streamlit Cloud)
    # Uses Groq for fast, free inference of Llama models
    LLM_MODEL_NAME = "llama-3.1-8b-instant"
    OLLAMA_BASE_URL = None
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 2. Embedding Settings
# We use the same embedding model for both (it runs on CPU)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# 3. Retrieval Settings
RETRIEVAL_K = 4