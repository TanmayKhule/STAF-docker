from typing import Dict, List, Optional
from langchain.schema import Document
from app.services.llm_service import LLMService
from app.models.schemas import TestCase
import json
import time
import logging
from langchain.output_parsers import PydanticOutputParser
from app.models.schemas import TestCaseSet
from app.core.config import config

logger = logging.getLogger(__name__)

class TestCaseGenerator:
    def __init__(self):
        self.llm_service = LLMService.get_instance()
        self.max_regeneration_attempts = config["llm"]["max_attempts"]
        self.output_parser = PydanticOutputParser(pydantic_object=TestCaseSet)




    def _parse_test_cases(self, response: str) -> List[TestCase]:
        """Parse and validate generated test cases with improved error handling."""
        try:
            logger.info("Starting test case parsing")
            
            # Handle string response
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                    logger.info("Successfully parsed JSON string response")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse response as JSON: {e}")
                    logger.error(f"Raw response: {response[:200]}...")
                    raise ValueError("Invalid JSON response from LLM")

            # Validate response structure
            if not isinstance(response, dict):
                logger.error(f"Response is not a dictionary: {type(response)}")
                raise ValueError("Invalid response format")

            test_cases_data = response.get("test_cases", [])
            if not test_cases_data:
                logger.error("No test cases found in response")
                raise ValueError("No test cases generated")

            logger.info(f"Found {len(test_cases_data)} test cases to parse")

            test_cases = []
            for i, tc_data in enumerate(test_cases_data):
                try:
                    logger.info(f"Processing test case {i+1}")
                    
                    # Convert list fields to strings
                    for field in ["setup", "test_code", "teardown"]:
                        if isinstance(tc_data.get(field), list):
                            tc_data[field] = "\n".join(tc_data[field])

                    # Validate required fields
                    missing_fields = [
                        field for field in TestCase.__annotations__ 
                        if field not in tc_data
                    ]
                    if missing_fields:
                        logger.error(f"Test case {i+1} missing required fields: {missing_fields}")
                        continue

                    test_case = TestCase(**tc_data)
                    test_cases.append(test_case)
                    logger.info(f"Successfully parsed test case {i+1}: {test_case.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to parse test case {i+1}: {e}")
                    logger.error(f"Test case data: {tc_data}")
                    continue

            if not test_cases:
                logger.error("No valid test cases were parsed")
                raise ValueError("Failed to parse any valid test cases")

            logger.info(f"Successfully parsed {len(test_cases)} test cases")
            return test_cases

        except Exception as e:
            logger.error(f"Test case parsing failed: {str(e)}")
            logger.exception("Detailed traceback:")
            raise

    async def generate_test_cases(
        self, 
        attack_tree: Dict, 
        analysis: Dict, 
        documents: List[Document]
    ) -> Dict[int, TestCase]:
        """Generate initial test cases and return as a dictionary keyed by ID."""
        try:
            logger.info("Starting test case generation")
            
            # Create the generation prompt
            prompt = self._create_generation_prompt(attack_tree, analysis, documents)
            
            # Generate test cases
            response = await self.llm_service.generate(prompt)
            
            # Parse and validate the test cases
            test_cases_dict = {}
            raw_test_cases = response.get("test_cases", [])
            
            for tc_data in raw_test_cases:
                try:
                    # Ensure each test case has an ID
                    if "id" not in tc_data:
                        tc_data["id"] = len(test_cases_dict) + 1
                        
                    # Create TestCase object and store in dictionary
                    test_case = TestCase(**tc_data)
                    test_cases_dict[test_case.id] = test_case
                    logger.info(f"Generated test case {test_case.id}: {test_case.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to parse test case: {e}")
                    continue
                    
            if not test_cases_dict:
                raise ValueError("No valid test cases were generated")
                
            return test_cases_dict
            
        except Exception as e:
            logger.error(f"Test case generation failed: {str(e)}")
            raise

    async def regenerate_test_cases(
        self,
        existing_test_cases: Dict[int, TestCase],
        missing_vulnerabilities: List[str],
        improvement_suggestions: List[Dict]
    ) -> Dict[int, TestCase]:
        """
        Regenerate test cases while maintaining all existing cases and properly merging improvements.
        """
        try:
            logger.info(f"Starting regeneration with {len(existing_test_cases)} existing test cases")
            regeneration_attempts = config["llm"]["max_attempts"]
            
            # Get the next available ID
            max_id = max(existing_test_cases.keys(), default=0)
            
            # Create mapping of test case IDs to their suggestions
            improvement_map = {
                suggestion.get("test_case_id"): suggestion.get("suggestions", [])
                for suggestion in improvement_suggestions
                if "test_case_id" in suggestion
            }
            
            logger.info(f"Created improvement map for {len(improvement_map)} test cases")
            
            # Create prompt for regeneration
            prompt = self._create_regeneration_prompt(
                existing_test_cases,
                missing_vulnerabilities,
                improvement_suggestions
            )
            
            # Generate new/improved test cases
            response = await self.llm_service.generate(prompt)
            new_test_cases = self._parse_test_cases(response)
            
            # Create a working copy of existing test cases
            final_test_cases = existing_test_cases.copy()
            
            # Process each new/improved test case
            for new_case in new_test_cases:
                existing_id = None
                
                # Try to match with existing test case
                for test_id, existing_case in existing_test_cases.items():
                    if (existing_case.name == new_case.name or 
                        existing_case.vulnerability_addressed == new_case.vulnerability_addressed):
                        existing_id = test_id
                        break
                
                if existing_id:
                    # Update existing test case while preserving its ID
                    new_case.id = existing_id
                    final_test_cases[existing_id] = new_case
                    logger.info(f"Updated existing test case {existing_id}: {new_case.name}")
                else:
                    # Add as new test case with next available ID
                    max_id += 1
                    new_case.id = max_id
                    new_case.name = f"Test Case {max_id}: {new_case.name}"
                    final_test_cases[max_id] = new_case
                    logger.info(f"Added new test case {max_id}: {new_case.name}")
            
            # Verify we haven't lost any test cases
            logger.info(f"Original test cases: {len(existing_test_cases)}")
            logger.info(f"Final test cases: {len(final_test_cases)}")
            
            return final_test_cases
                
        except Exception as e:
            logger.error(f"Test case regeneration failed: {str(e)}")
            logger.exception("Detailed traceback:")
            # Return original test cases on error
            return existing_test_cases



    async def check_test_cases(
        self, 
        attack_tree: Dict, 
        test_cases: Dict[int, TestCase]
    ) -> Dict:
        """Check test cases with enhanced logging."""
        logger.info("=== STARTING TEST CASE QUALITY CHECK ===")
        start_time = time.time()
        
        try:
            logger.info(f"Processing {len(test_cases)} test cases for quality check")
            
            # Log test cases being checked
            for tc_id, tc in test_cases.items():
                logger.info(f"Processing test case {tc_id}: {tc.name}")
            
            # Create and log prompt length
            prompt = self._create_check_prompt(attack_tree, test_cases)
            logger.info(f"Created check prompt with length: {len(prompt['content'])}")
            
            # Log LLM request start
            logger.info("Sending request to LLM service...")
            llm_start = time.time()
            response = await self.llm_service.generate(prompt)
            logger.info(f"LLM response received in {time.time() - llm_start:.2f} seconds")
            
            # Log response validation
            if isinstance(response, dict):
                logger.info("Successfully received structured response from LLM")
                logger.info(f"Response keys: {list(response.keys())}")
            else:
                logger.warning(f"Unexpected response type: {type(response)}")
                logger.debug(f"Raw response: {response}")
                
            duration = time.time() - start_time
            logger.info(f"=== QUALITY CHECK COMPLETED IN {duration:.2f} SECONDS ===")
            return response
            
        except Exception as e:
            logger.error("=== QUALITY CHECK FAILED ===")
            logger.error(f"Error details: {str(e)}")
            logger.exception("Full traceback:")
            raise



    def _create_generation_prompt(
        self, 
        attack_tree: Dict, 
        analysis: Dict, 
        documents: List[Document]
    ) -> Dict[str, str]:
        """Create prompt for initial test case generation."""
        try:
            # Validate inputs
            if not isinstance(attack_tree, dict):
                raise ValueError(f"Attack tree must be a dict, got {type(attack_tree)}")
            if not isinstance(analysis, dict):
                raise ValueError(f"Analysis must be a dict, got {type(analysis)}")
            if not isinstance(documents, list):
                raise ValueError(f"Documents must be a list, got {type(documents)}")

            # Create formatted document content with length limits
            doc_content = self._format_documents(documents)

            prompt = {
            "role": "user",
            "content": f"""You are an elite security test engineer with extensive experience in creating comprehensive, robust, and detailed Python test suites, specifically targeting vulnerabilities in diverse systems and infrastructures. Your task is to generate an exceptionally thorough set of Python test cases for a system based on a detailed attack tree analysis and additional context documents. The goal is to create a test suite that rigorously validates the security of the system and defends against ALL identified vulnerabilities, no matter how complex or nuanced.

Attack Tree:
{json.dumps(attack_tree, indent=2)}

Analysis:
{json.dumps(analysis, indent=2)}

Reference Documents:
{doc_content}


### Critical Requirements for the Test Suite:
1. **Vulnerability Focus**: Each test case must be designed to specifically test for vulnerabilities mentioned in the attack tree, focusing on real security issues like improper access control, misconfigurations, insecure data storage, unencrypted data transmission, excessive permissions, and API abuse. Go beyond availability checks to test for security weaknesses.
2. **Alignment with Attack Tree**: Create a separate test case for each attack vector identified in the attack tree analysis. No vulnerability should be left untested. Ensure test cases mirror the attack vectors, addressing both the specific scenarios and potential bypass methods.
3. **Depth in Vulnerability Testing**: For complex vulnerabilities, create multiple test methods to cover various scenarios, edge cases, and potential bypass methods, ensuring thorough coverage. Incorporate multiple stages of attacks (e.g., privilege escalation, lateral movement, denial of service).
4. **Realistic Attack Scenarios**: Use realistic, diverse, and complex data sets in your tests, simulating real-world usage and attack scenarios relevant to the system under test. The test cases should go beyond basic functionality checks to simulate attack patterns like privilege escalation, improper role assignment, and compromised credentials.
5. **Security Assertions**: Implement sophisticated assertions that not only verify functionality but also confirm the absence of security vulnerabilities (e.g., improper access permissions, unencrypted data storage, misconfigured security settings).
6. **System-Specific Vulnerability Testing**: Ensure test cases cover security aspects of the specific components or services of the system, focusing on vulnerability testing rather than just availability. For instance, test improper access control for data storage, unencrypted communication channels, insecure network configurations, etc.
7. **Positive and Negative Tests**: Implement both positive tests (verifying secure configurations and expected behavior) and negative tests (attempting to exploit vulnerabilities) for each vulnerability.
8. **Parameterized Tests**: Where applicable, implement parameterized tests to cover a wide range of inputs efficiently, ensuring scalability in vulnerability testing.
9. **Race Conditions and Timing Attacks**: Include tests for complex vulnerabilities like race conditions, timing attacks, and multi-stage attacks that simulate real-world advanced persistent threat (APT) scenarios.
10. **Error Handling**: Include error-handling tests to ensure that the system behaves securely under various error conditions, such as handling malformed requests or insufficient permissions securely.
11. **Comprehensive Coverage**: Maintain comprehensive coverage of all system components but ensure that the tests are focused on testing security vulnerabilities (e.g., roles with excessive permissions, unencrypted data storage, misconfigured security alerts that fail to detect incidents).
12. **Thorough Documentation**: Add exhaustive comments and docstrings explaining:
    - The purpose of each test method
    - The specific vulnerability or attack vector being tested
    - The expected outcome and why it's secure
    - Any subtle points or non-obvious security implications
13. **Runnable Code**: Provide Python code for each test case that is immediately runnable. Ensure the use of the `unittest` framework and appropriate mocking techniques where applicable, but move beyond basic mocking to simulate real-world attack scenarios.
14. **Coverage Report**: After generating the test cases, provide a coverage report explaining how each identified vulnerability is addressed by the test cases.

Example Output Format:
{{
    "test_cases": [
        {{
            "id": 1,
            "name": "Test Authentication Bypass via SQL Injection",
            "description": "Verifies system resistance to SQL injection attempts",
            "vulnerability_addressed": "SQL Injection",
            "setup": "def setup():\\n    self.db = TestDatabase()\\n    self.user = TestUser()",
            "test_code": "def test_sql_injection():\\n    payload = \\\"' OR '1'='1\\\"\\n    assert not self.db.authenticate(payload)",
            "teardown": "def teardown():\\n    self.db.cleanup()\\n    self.user.remove()",
            "expected_result": "Authentication attempt with SQL injection payload is blocked"
        }}
    ]
}}
                                                    
Each test case should include an 'id' field that is a unique integer.
                                                    
{self.output_parser.get_format_instructions()}

Remember to provide actual Python code for each test case, not just descriptions, and ensure ALL vulnerabilities are covered.
"""        }
        
            logger.info(f"Created generation prompt with length: {len(prompt['content'])}")
            return prompt

        except Exception as e:
            logger.error(f"Failed to create generation prompt: {e}")
            raise


    def _create_regeneration_prompt(
        self,
        existing_test_cases: Dict[int, Dict],
        missing_vulnerabilities: List[str],
        improvement_suggestions: List[Dict]
    ) -> Dict[str, str]:
        """Create prompt for test case regeneration with proper JSON serialization."""
        try:
            # Convert TestCase objects to dictionaries
            serializable_test_cases = {}
            for tc_id, tc in existing_test_cases.items():
                if isinstance(tc, TestCase):
                    # Use the Pydantic model's dict() method
                    serializable_test_cases[tc_id] = tc.dict()
                else:
                    # If it's already a dict, use it as is
                    serializable_test_cases[tc_id] = tc
            
            # Now serialize to JSON
            test_cases_json = json.dumps(serializable_test_cases, indent=2)
            missing_vulns_json = json.dumps(missing_vulnerabilities, indent=2)
            suggestions_json = json.dumps(improvement_suggestions, indent=2)
            return {
                "role": "user",
                "content": f"""You are an elite security test engineer with extensive experience in creating comprehensive and robust test suites across various systems and infrastructures. Your critical task is to **modify the existing test cases** based on the improvement suggestions provided, and **add new test cases** for any missing vulnerabilities.

    ### Existing Test Cases:
    {test_cases_json}

    ### Improvement Suggestions:
    {suggestions_json}

    ### Missing Vulnerabilities:
    {missing_vulns_json}

    ### Instructions:
    1. **Modify the existing test cases** to incorporate all improvement suggestions. Only make changes where improvements are suggested; retain other content.
    2. For each **missing vulnerability**, **create a new test case** that exactly addresses the vulnerability.
    3. Ensure that all test cases use appropriate and actual code relevant to the system under test, utilizing standard libraries or APIs suitable for that system.
    4. Include all necessary **setup**, including required imports and initialization of system components or services if needed.
    5. The test code must be **complete, runnable Python code**. Do not use pseudocode or placeholders.
    6. Follow **best practices** for the system or domain you are testing, and use appropriate methods and calls.
    7. Each test case should demonstrate both the **vulnerable state and the secure state**.
    8. Use **assert statements** to clearly indicate what constitutes a pass or fail condition.
    9. Each test case should include an 'id' field that is a unique integer.
    {self.output_parser.get_format_instructions()}
    Example Output Format:
    {{
        "test_cases": [
            {{
                "id": 1,
                "name": "Test Authentication Bypass via SQL Injection",
                "description": "Verifies system resistance to SQL injection attempts",
                "vulnerability_addressed": "SQL Injection",
                "setup": "def setup():\\n    self.db = TestDatabase()\\n    self.user = TestUser()",
                "test_code": "def test_sql_injection():\\n    payload = \\\"' OR '1'='1\\\"\\n    assert not self.db.authenticate(payload)",
                "teardown": "def teardown():\\n    self.db.cleanup()\\n    self.user.remove()",
                "expected_result": "Authentication attempt with SQL injection payload is blocked"
            }}
        ]
    }}

    """}
        
        except Exception as e:
            logger.error(f"Failed to create regeneration prompt: {e}")
            raise

    def _create_check_prompt(
        self, 
        attack_tree: Dict, 
        test_cases: Dict[int, TestCase]
    ) -> Dict[str, str]:
        """Create prompt for test case evaluation with proper serialization."""
        logger.info("Creating check prompt with proper TestCase serialization")
        
        try:
            # Convert TestCase objects to dictionaries first
            serializable_test_cases = {}
            for tc_id, tc in test_cases.items():
                if isinstance(tc, TestCase):
                    # Use the model's built-in dict() method from Pydantic
                    serializable_test_cases[tc_id] = tc.dict()
                else:
                    # If it's already a dict, use it as is
                    serializable_test_cases[tc_id] = tc
                    
            logger.info(f"Successfully serialized {len(serializable_test_cases)} test cases")
        
        except Exception as e:
            logger.error(f"Failed to create check prompt: {e}")
            logger.exception("Detailed traceback:")
            raise


        return  {
            "role": "user",
            "content": f"""You are an expert security analyst with a critical eye for detail. Your task is to rigorously check if the generated test cases align with the attack tree, are complete, and are of high quality.

    Attack Tree:
    {json.dumps(attack_tree, indent=2)}


    Generated Test Cases:
    {json.dumps(serializable_test_cases, indent=2)}

    Perform a thorough analysis of the test cases and provide the following:
    1. Alignment: Are the test cases properly aligned with the vulnerabilities identified in the attack tree? Be extremely critical.
    2. Completeness: Do the test cases cover ALL vulnerabilities mentioned in the attack tree? List any that are missing or inadequately covered.
    3. Runnability: Is the code in the test cases runnable in Python? Are there any missing imports, setup steps, or other issues that would prevent immediate execution?
    4. Quality: Assess the quality of each test case. Are they thorough? Do they actually test what they claim to test?
    5. Improvements: Suggest specific, detailed improvements for each test case that falls short in any way.

    Provide your analysis in a structured JSON format with the following keys:
    - alignment_score (0-100, be very strict)
    - completeness_score (0-100, be very strict)
    - runnability_score (0-100, be very strict)
    - quality_score (0-100, be very strict)
    - missing_vulnerabilities (list of vulnerabilities not covered or inadequately covered)
    - improvement_suggestions (list of objects, each containing:
        - test_case_name: the name of the test case that needs improvement
        - suggestions: list of specific, detailed suggestions for improving that test case)

    Be extremely critical in your assessment. We need to ensure these test cases are of the highest possible quality.
    Provide a JSON evaluation with these exact fields:
    {{
        "alignment_score": <0-100>,      // How well tests align with vulnerabilities
        "completeness_score": <0-100>,   // Coverage of all attack vectors
        "quality_score": <0-100>,        // Overall test implementation quality
        "runnability_score": <0-100>,    // How well tests can be executed
        "missing_vulnerabilities": [      // List of uncovered vulnerabilities
            "vulnerability name"
        ],
        "improvement_suggestions": [      // Specific improvements needed
            {{
                "test_case_id": <id>,
                "suggestions": ["specific improvement needed"]
            }}
        ]
    }}

    Focus on:
    1. Security testing coverage
    2. Test implementation quality
    3. Missing attack vectors
    4. Specific improvements needed
    """
        }

    def _parse_test_cases(self, response: str) -> List[TestCase]:
        """Parse and validate generated test cases."""
        try:
            if not response:
                logger.error("Empty response received")
                raise ValueError("Empty response received")
                
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.debug(f"Raw response: {response}")
                    raise ValueError("Invalid JSON response")
                    
            test_cases = response.get("test_cases", [])
            if not test_cases:
                logger.error("No test cases found in response")
                raise ValueError("No test cases found in response")
                
            parsed_cases = []
            for tc_data in test_cases:
                if not all(key in tc_data for key in TestCase.__annotations__):
                    logger.error(f"Invalid test case structure: {tc_data}")
                    continue
                try:
                    parsed_cases.append(TestCase(**tc_data))
                except Exception as e:
                    logger.error(f"Failed to parse test case: {e}")
                    continue
                    
            if not parsed_cases:
                raise ValueError("No valid test cases could be parsed")
                
            return parsed_cases
            
        except Exception as e:
            logger.error(f"Test case parsing failed: {str(e)}")
            raise ValueError("No test cases generated")


    def _parse_check_response(self, response: Dict) -> Dict:
        """Parse and validate test case evaluation response."""
        required_fields = [
            "alignment_score", "completeness_score", 
            "runnability_score", "quality_score",
            "missing_vulnerabilities", "improvement_suggestions"
        ]
        
        if not all(field in response for field in required_fields):
            raise ValueError("Invalid check response format")
            
        return response

    def _format_documents(self, documents: List[Document]) -> str:
        """Format reference documents for prompt."""
        return "\n\n".join(
            f"Document {i+1}:\n{doc.page_content}" 
            for i, doc in enumerate(documents)
        )
    
    