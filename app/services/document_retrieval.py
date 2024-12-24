from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List, Dict
from app.core.config import config
import torch
import os
import logging
import time
from huggingface_hub import login
import json

logger = logging.getLogger(__name__)

class DocumentRetriever:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = config['huggingface']['model_name']
        self.num_rel = config['vectorstore']['num_docs']
        # Login to HuggingFace
        self._huggingface_login()
        
        self.embeddings = self._initialize_embeddings()
        self.vectorstore = self._init_vectorstore()
    
    def _huggingface_login(self):
        """Login to HuggingFace using token from config"""
        try:
            token = config['huggingface'].get('api_token')
            if not token:
                logger.warning("No HuggingFace API token found in config")
                return
                
            login(token=token)
            logger.info("Successfully logged in to HuggingFace")
        except Exception as e:
            logger.error(f"Failed to login to HuggingFace: {e}")
            raise

    def _initialize_embeddings(self):
        """Initialize embeddings model exactly as in notebook"""
        try:
            return HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={
                    "device": self.device,
                    "trust_remote_code": True
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                    "device": self.device
                }
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise


    def _init_vectorstore(self):
        """Initialize vectorstore exactly as in notebook"""
        try:
            return Chroma(
                persist_directory=config["vectorstore"]["persistent_dir"],
                embedding_function=self.embeddings,
                collection_name=config["vectorstore"]["collection_name"]
            )
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise



    # Your exact query function from the notebook
    def query_vectorstore(self, query: str, k: int = 5) -> List[Document]:
        """Query vectorstore with same implementation as notebook"""
        try:
            results = self.vectorstore.similarity_search(query, k=self.num_rel)
            logger.info(f"\nQuery Results:")
            for i, doc in enumerate(results, 1):
                logger.info(f"\nResult {i}:")
                logger.info(f"Content: {doc.page_content[:200]}...")
                logger.info(f"Metadata: {doc.metadata}")
                logger.info("-" * 80)
            return results
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            return []

    # Added for FastAPI compatibility but calls your original function
    async def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Async wrapper for compatibility"""
        return self.query_vectorstore(query, self.num_rel)


    def _create_grading_prompt(self, documents: List[Document], vulnerabilities: List[str]) -> Dict[str, str]:
        return {
            "role": "user",
            "content": f"""You are a security documentation expert. Grade these documents based on their relevance to the identified vulnerabilities.

Documents to grade:
{json.dumps([{"content": doc.page_content, "metadata": doc.metadata} for doc in documents], indent=2)}

Vulnerabilities to check:
{json.dumps(vulnerabilities, indent=2)}

For each document, provide a grade from 0-100 based on:
1. Direct relevance to vulnerabilities (40%)
2. Technical depth and accuracy (30%)
3. Actionability of information (30%)

Provide response in JSON format:
{{
    "grades": [
        {{
            "document_id": "doc_id",
            "grade": 85,
            "reasoning": "Detailed explanation"
        }}
    ]
}}"""
        }
