📘 IMF Document Intelligence: Local Multi-Modal RAG

🔗 Live Deployed App: https://multi-modal-rag-system-project.streamlit.app/

A professional, fully local Retrieval-Augmented Generation (RAG) system designed to analyze complex financial documents (like IMF Article IV reports) with high precision.

Built for accuracy, privacy, and low-resource environments.

🚀 Overview

This project implements an end-to-end Document Intelligence pipeline that ingests PDF reports, preserves complex tabular structures, and allows users to ask questions via a modern, "Glassmorphism" UI.

Unlike standard RAG tutorials that rely on paid APIs (OpenAI/Anthropic), this system runs 100% locally using Ollama and HuggingFace, making it free to run and privacy-compliant. It is optimized to run on consumer hardware with a storage footprint under 3GB.

✨ Key Features

🔐 Fully Local & Private: No API keys required. Runs entirely on your CPU/GPU.

📊 Smart Table Parsing: Uses pdfplumber to convert financial tables into Markdown, allowing the LLM to "read" rows and columns accurately.

🧠 Context-Aware Citations: Every answer cites specific Page Numbers and provides the raw source snippet for verification.

🎨 Premium UI: A polished Streamlit interface with dark mode, clear typography, and "Kid-Friendly" explanations for complex financial terms.

⚡ Optimized Pipeline: Intelligent caching skips redundant processing steps (Ingestion/Indexing) to save time during development.

🛠️ Tech Stack

LLM: Ollama running llama3.2:1b (Lightweight & Fast)

Embeddings: sentence-transformers/all-MiniLM-L6-v2 (via HuggingFace)

Vector Database: FAISS (CPU)

Orchestration: LangChain (LCEL)

PDF Processing: pdfplumber (for table fidelity)

Frontend: Streamlit with custom CSS

⚙️ Installation

1. Prerequisites

You must have Ollama installed and running.

Install Ollama.

Pull the required lightweight model (1.3GB):

ollama pull llama3.2:1b


2. Clone & Setup

# Clone the repository
git clone [https://github.com/yourusername/Big-Air-Lab-AI-ML-Intern-Assignment.git](https://github.com/yourusername/Big-Air-Lab-AI-ML-Intern-Assignment.git)
cd imf-document-intelligence

# Create a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt


3. Prepare Data

Place your target PDF file (e.g., qatar_test_doc.pdf) in the raw data folder:

data/raw/qatar_test_doc.pdf


(Note: The directory structure is created automatically on the first run if it doesn't exist.)

🏃 Usage

Option A: One-Click Pipeline (Recommended)

Run the entire ingestion, indexing, and testing process in one command:

python run_pipeline.py


Command Line Arguments:

--force: Forces re-ingestion of the PDF and re-creation of the vector index (useful if you changed the PDF).

--skip-ingest: Skips the heavy PDF processing and runs only the QA inference (useful for quick testing).

--query "Your question": Ask a specific question from the command line.

Example:

python run_pipeline.py --skip-ingest --query "What are the fiscal risks for 2024?"


Option B: The User Interface

Launch the web application to interact with the document:

streamlit run app.py


The app provides a chat interface where you can:

Ask natural language questions.

View the exact Page Number and Source Text for every claim.

See simplified, "ELI5" (Explain Like I'm 5) answers for complex topics.

📂 Project Structure

.
├── config.py               # Central configuration (Paths, Model names)
├── document_processor.py   # Parsing logic (PDF -> Markdown/JSON chunks)
├── vector_store.py         # Embedding generation & FAISS management
├── llm_qa.py               # RAG Logic (Ollama connection, Prompt templates)
├── run_pipeline.py         # CLI Orchestrator for the whole workflow
├── app.py                  # Streamlit UI
├── requirements.txt        # Dependencies
└── data/                   # Local artifacts (Ignored by Git)
    ├── raw/                # Input PDFs
    ├── processed/          # JSON chunks
    ├── images/             # Extracted images
    └── vector_store/       # FAISS index files


🧩 Architecture Decisions

Why Local?

To meet strict data privacy requirements and eliminate operational costs, this system uses quantized local models. llama3.2:1b was chosen for its surprising reasoning capability relative to its tiny size (1.3GB), fitting easily into limited environments (e.g., laptops with <8GB RAM).

Why Markdown for Tables?

Standard PDF parsers flatten text, destroying the structure of financial tables (e.g., Balance Sheets). By detecting tables and converting them to Markdown format (| Revenue | 2024 |), we preserve the row-column relationships, allowing the small LLM to answer queries like "What was the revenue in 2024?" accurately.

📜 License

This project is open-source and available under the MIT License.
