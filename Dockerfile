# 1. Base Image: Lightweight Python 3.12
FROM python:3.12-slim

# 2. Set Environment Variables
# Prevents Python from buffering stdout/stderr (logs appear immediately)
ENV PYTHONUNBUFFERED=1
# OLLAMA_HOST needs to be exposed for local communication
ENV OLLAMA_HOST=0.0.0.0

# 3. Install System Dependencies
# We need 'curl' to download Ollama
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Ollama (The AI Engine)
RUN curl -fsSL https://ollama.com/install.sh | sh

# 5. Set Working Directory
WORKDIR /app

# 6. Install Python Dependencies
COPY requirements.txt .
# We use --no-cache-dir to keep the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy Application Code
COPY . .

# 8. Expose Ports
# 8501 = Streamlit
EXPOSE 8501

# 9. Entrypoint
# We use a script to start both Ollama and Streamlit together
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]