import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# MODEL CONFIGURATION
# ==========================================

# Toggle this for Cloud Deployment
# Options: "local" (Ollama), "cloud" (Groq/OpenAI)
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "local")

# 1. LLM Settings
if DEPLOYMENT_MODE == "local":
    LLM_MODEL_NAME = "llama3.2:1b"
    OLLAMA_BASE_URL = "http://localhost:11434"
else:
    # For Streamlit Cloud, we use Groq (Free Tier available)
    # It runs Llama models incredibly fast in the cloud
    LLM_MODEL_NAME = "llama-3.1-8b-instant"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
