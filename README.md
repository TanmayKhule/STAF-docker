# Security Testing Automation Framework (STAF)

A framework that generates security test cases from attack trees using LLMs.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your Ollama models:
   * Install Ollama and pull your desired models
   * The model names in config.yaml must match exactly with your Ollama model names
   * Example: If you pulled "mistral" and "mistral-small", use those exact names

3. Set up ChromaDB:
   * Download the pre-built ChromaDB from Google Drive [link]
   * Extract the contents to `./chromadb` directory in your project root
   * This contains pre-indexed security documentation for better test generation

4. Create `config.yaml` in the root directory:
   ```yaml
   llm:
     model: "llama3.1"  # Must match your Ollama model name
     grader_model: "llama3.1"  # Must match your Ollama model name
     models_to_pull: ["llama3.1", "llama3.1"]  # List all models to pull at startup
     temperature: 0
     max_tokens: 200000
     max_attempts: 3

   vectorstore:
     persistent_dir: "./chromadb"
     collection_name: "security_docs"
     num_docs: 5

   huggingface:
     model_name: "encoder model name"
     api_token: "your_huggingface_token"  # Required! Add your HuggingFace token
   ```

5. Required API Keys:
   * HuggingFace token: Get from [HuggingFace](https://huggingface.co/)
   * Add ALL tokens to config.yaml in their respective sections
   * The application won't work without proper API keys configured

## Docker Deployment

You can also run STAF using our pre-built Docker image:

```bash
# Pull the image
docker pull yourusername/staf:latest

# Run the container
docker run -p 8000:8000 yourusername/staf:latest
```

The Docker image includes all necessary dependencies and the pre-built ChromaDB.

## Running the API

Start the server:
```bash
uvicorn app.main:app --reload
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
  "k": 5  // number of results
}
```

View the API documentation at `http://localhost:8000/docs`
