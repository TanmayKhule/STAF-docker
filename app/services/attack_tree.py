from typing import Dict, List
from pydantic import BaseModel, Field
from app.services.llm_service import LLMService
import logging
from app.models.schemas import DocumentRetrievalPrompt
import json
from langchain.output_parsers import PydanticOutputParser


class AttackTreeAnalyzer:
    def __init__(self):
        self.llm_service = LLMService.get_instance()
        self.logger = logging.getLogger(__name__)  # Get configured logger
        self.logger.info("AttackTreeAnalyzer initialized")  # Test log message


    async def analyze(self, attack_tree: Dict) -> DocumentRetrievalPrompt:
        """
        Analyze attack tree to identify vulnerabilities and generate retrieval query.
        """
        try:
            prompt = self._create_analysis_prompt(attack_tree)
            raw_response = await self.llm_service.generate(prompt)

            
            # Handle different response types
            structured_data = self._process_raw_response(raw_response)
            
            # Create and validate DocumentRetrievalPrompt
            result = DocumentRetrievalPrompt(
                keywords=structured_data["keywords"],
                vulnerabilities=structured_data["vulnerabilities"],
                query=structured_data["query"]
            )
            
            # Log successful analysis
            self.logger.info(f"Successfully analyzed attack tree: {len(result.vulnerabilities)} vulnerabilities found")
            return result
            
        except Exception as e:
            self.logger.error(f"Attack tree analysis failed: {str(e)}")
            # Return valid default response instead of raising
            return DocumentRetrievalPrompt(
                keywords=["security", "vulnerability"],
                vulnerabilities=[{
                    "name": "Unknown",
                    "description": "Analysis failed to complete",
                    "severity": "Unknown"
                }],
                query="security vulnerability testing"
            )

    def _process_raw_response(self, response: Dict) -> Dict:
        """Process and validate raw LLM response."""
        try:
            # Initialize default structure
            processed_data = {
                "keywords": [],
                "vulnerabilities": [],
                "query": ""
            }
            
            if isinstance(response, dict):
                # Process keywords
                if "keywords" in response and isinstance(response["keywords"], list):
                    processed_data["keywords"] = [
                        str(k) for k in response["keywords"] if k is not None
                    ]
                
                # Process vulnerabilities
                if "vulnerabilities" in response and isinstance(response["vulnerabilities"], list):
                    processed_data["vulnerabilities"] = [
                        self._validate_vulnerability(v) for v in response["vulnerabilities"]
                        if isinstance(v, dict)
                    ]
                
                # Process query
                if "query" in response and isinstance(response["query"], str):
                    processed_data["query"] = response["query"]
                
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing LLM response: {str(e)}")
            return {
                "keywords": ["error", "processing", "failed"],
                "vulnerabilities": [{
                    "name": "Processing Error",
                    "description": "Failed to process LLM response",
                    "severity": "Unknown"
                }],
                "query": "security vulnerability analysis"
            }

    def _validate_vulnerability(self, vuln: Dict) -> Dict:
        """Validate and format a vulnerability entry."""
        return {
            "name": str(vuln.get("name", "Unnamed Vulnerability")),
            "description": str(vuln.get("description", "No description provided")),
            "severity": str(vuln.get("severity", "Unknown"))
        }

    def _create_analysis_prompt(self, attack_tree: Dict) -> Dict[str, str]:
        """Create structured prompt for attack tree analysis."""
        output_parser = PydanticOutputParser(pydantic_object=DocumentRetrievalPrompt)
        return {
            "role": "user",
            "content": f"""You are a senior security engineer tasked with performing an in-depth analysis of an attack tree to identify all possible vulnerabilities in the system. Your analysis should be thorough and comprehensive, leaving no stone unturned.

Here is the attack tree in JSON format:

{attack_tree}

Please perform the following tasks:
1. Conduct a detailed, step-by-step analysis of the attack tree, considering all possible attack vectors and their implications.
2. Identify and Describe all potential vulnerabilities in the system, including their severity, potential impact, and possible mitigation strategies.
3. Extract relevant keywords from the attack tree that are crucial for understanding the system's security landscape.
4. Create a comprehensive query for document retrieval that covers all aspects of the attack tree and identified vulnerabilities.

Ensure your analysis is exhaustive and doesn't overlook any potential security risks. Consider both obvious and non-obvious attack paths.

{output_parser.get_format_instructions()}

Provide your in-depth analysis and the document retrieval prompt in the specified JSON format.
"""
        }