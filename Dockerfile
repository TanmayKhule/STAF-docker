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

# Copy application code
COPY . .

# Create startup script with fixed process management
COPY <<EOF /start-ollama.sh
#!/bin/bash
set -x  # Enable debug mode

echo "Starting Ollama service..."
# Start Ollama in the background and redirect output
ollama serve > /var/log/ollama.log 2>&1 &

# Wait for Ollama to start listening on its port
echo "Waiting for Ollama to become available..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:11434/api/version > /dev/null; then
        echo "Ollama is ready!"
        # Pull required models
        for model in llama3.1:70b llama3.1:8b llama3.2:1b; do
            echo "Pulling model: \$model"
            ollama pull \$model
        done
        echo "Starting FastAPI application..."
        exec "\$@"
        exit 0
    fi
    echo "Attempt \$i: Waiting for Ollama..."
    sleep 1
done

echo "Ollama failed to start. Latest logs:"
tail -n 50 /var/log/ollama.log
exit 1
EOF

RUN chmod +x /start-ollama.sh

# Use the startup script as the entrypoint
ENTRYPOINT ["/start-ollama.sh"]

# Command to run the application
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]