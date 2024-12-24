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

3. Create `config.yaml` in the root directory:
   ```yaml
   llm:
     model: "mistral"  # Must match your Ollama model name
     grader_model: "mistral-small"  # Must match your Ollama model name
     models_to_pull: ["mistral", "mistral-small"]  # List all models to pull at startup
     temperature: 0
     max_tokens: 200000
     max_attempts: 3

   vectorstore:
     persistent_dir: "./chromadb"
     collection_name: "security_docs"
     num_docs: 5

   huggingface:
     model_name: "BAAI/bge-large-en-v1.5"
     api_token: "your_huggingface_token"  # Required! Add your HuggingFace token
   ```

4. Required API Keys:
   * HuggingFace token: Get from [HuggingFace](https://huggingface.co/)
   * Add ALL tokens to config.yaml in their respective sections
   * The application won't work without proper API keys configured

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
