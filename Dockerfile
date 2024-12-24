FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Configure apt and install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    git \
    ca-certificates \
    dnsutils \
    iputils-ping \
    net-tools \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/models \
    && mkdir -p /app/.cache/huggingface

# Set environment variables for HuggingFace
ENV HF_TOKEN=""
ENV HUGGINGFACE_TOKEN=""
ENV HF_HOME="/app/.cache/huggingface"
ENV TAVILY_API_KEY=""
ENV TRANSFORMERS_OFFLINE=0
ENV HF_DATASETS_OFFLINE=0
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Create the start script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting Ollama service..."\n\
\n\
# Start Ollama in the background\n\
echo "Waiting for Ollama to become available..."\n\
for i in {1..30}; do\n\
    if ollama serve & curl -s http://127.0.0.1:11434/api/version; then\n\
        echo "Ollama is ready!"\n\
        break\n\
    fi\n\
    sleep 1\n\
done\n\
\n\
# Pull required models\n\
models=$(python3 -c '\''\n\
import yaml\n\
with open("config.yaml") as f:\n\
    config = yaml.safe_load(f)\n\
models = config.get("llm", {}).get("models_to_pull", [])\n\
print(" ".join(models))\n\
'\'')\n\
\n\
for model in $models; do\n\
    echo "Pulling model: $model"\n\
    ollama pull $model\n\
done\n\
\n\
# Start FastAPI application\n\
echo "Starting FastAPI application..."\n\
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 80' > /app/start-ollama.sh

# Make the script executable
RUN chmod +x /app/start-ollama.sh


# Copy application code
COPY . .


# Use the startup script as the entrypoint
ENTRYPOINT ["/app/start-ollama.sh"]

# Command to run the application
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
