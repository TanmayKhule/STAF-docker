# STAF (Security Testing Automation Framework)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11-green.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)

A framework that generates security test cases from attack trees using LLMs.

## Quick Start with Docker

The fastest way to get started is using our pre-built Docker image:

```bash
# Pull the image
docker pull yourusername/staf:latest

# Run with GPU support and host network
docker run --network=host --gpus=all --env-file ./.env -v ./custom_config.yaml:/app/config.yaml staf-api:latest
```

## Manual Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Ollama
* Install Ollama from [ollama.ai](https://ollama.ai)
* Pull the required model:
```bash
ollama pull llama2:8b
```

### 3. Set up ChromaDB
* Download the pre-built ChromaDB from [Google Drive](https://drive.google.com/file/d/1R9cARRnoTBbQzHGM49mHqNeLwC7cm9eD/view?usp=drive_link)
* Extract the contents to `./chromadb` directory in your project root
* This contains pre-indexed security documentation for better test generation

### 4. Configuration

Create `custom_config.yaml`:

```yaml
llm:
  provider: "ollama"
  model: "llama2:8b"
  max_attempts: 4
  grader_model: "llama2:8b"
  temperature: 0
  max_tokens: 200000
  timeout: 240
  models_to_pull:
    - "llama2:8b"

vectorstore:
  model_name: "Alibaba-NLP/gte-large-en-v1.5"
  collection_name: "vulnerabilities"
  persistent_dir: "./chromadb"
  num_docs: 5

tavily:
  api_key: "your_tavily_key"

huggingface:
  api_token: "your_huggingface_token"
  model_name: "Alibaba-NLP/gte-large-en-v1.5"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  datefmt: "%Y-%m-%d %H:%M:%S"
```

### 5. Required API Keys
* HuggingFace token: Get from [HuggingFace](https://huggingface.co/)
* Add tokens to your `.env` file:
```bash
HF_TOKEN=your_huggingface_token
HUGGINGFACE_TOKEN=your_huggingface_token
TAVILY_API_KEY=your_tavily_key
```

## Building the Docker Image

If you want to build the image yourself:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/staf.git
cd staf
```

2. Build the Docker image:
```bash
docker build -t staf-api:latest .
```

3. Push to Docker Hub (optional):
```bash
docker login
docker tag staf-api:latest yourusername/staf-api:latest
docker push yourusername/staf-api:latest
```

## API Usage

### Generate Test Cases
```bash
POST /api/v1/generate-tests

Request body:
{
  "attack_tree": {
    "vulnerability": "Authentication Bypass",
    "main_attack_vectors": [
      {
        "vector": "SQL Injection",
        "description": "Bypass authentication using SQL injection",
        "sub_vectors": [
          {
            "vector": "Basic SQL Injection",
            "steps": [
              "Input malformed SQL in login fields",
              "Attempt to bypass authentication logic"
            ]
          }
        ]
      }
    ]
  }
}
```

### Query Vector Store
```bash
POST /api/v1/query-vectorstore

Request body:
{
  "query": "your search query",
  "k": 5  # number of results
}
```

### Health Check
```bash
GET /health
```

## API Documentation
Once the server is running, view the interactive API documentation at:
* Swagger UI: `http://localhost:80/docs`
* ReDoc: `http://localhost:80/redoc`

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
