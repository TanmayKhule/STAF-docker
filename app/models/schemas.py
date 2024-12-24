from pydantic import BaseModel
from langchain.schema import Document
from pydantic import BaseModel as LangchainBaseModel, Field
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any

class DocumentRetrievalPrompt(BaseModel):
    keywords: List[str] = Field(description="List of keywords extracted from the attack tree analysis")
    vulnerabilities: List[dict] = Field(description="List of identified vulnerabilities with details")
    query: str = Field(description="A comprehensive query for document retrieval")

class SubVector(BaseModel):
    vector: str
    steps: List[str]

class MainAttackVector(BaseModel):
    vector: str
    description: str
    sub_vectors: List[SubVector]

class AttackTree(BaseModel):
    vulnerability: str
    main_attack_vectors: List[MainAttackVector]

class AttackTreeRequest(BaseModel):
    attack_tree: AttackTree

    
class AttackTreeAnalysisResult(BaseModel):
    keywords: List[str] = Field(description="List of keywords extracted from the attack tree analysis")
    vulnerabilities: List[dict] = Field(description="List of identified vulnerabilities with details")
    query: str = Field(description="A comprehensive query for document retrieval")


class TestCase(BaseModel):
    id: int = Field(description="Unique identifier for the test case")
    name: str = Field(description="Name of the test case")
    description: str = Field(description="Detailed description of what the test case is checking")
    vulnerability_addressed: str = Field(description="The specific vulnerability this test case is addressing")
    setup: str = Field(description="Setup code for the test case")
    test_code: str = Field(description="Actual test code")
    teardown: str = Field(description="Teardown code for the test case")
    expected_result: str = Field(description="Expected result of the test")

    def to_dict(self) -> Dict[str, Any]:
        """Convert TestCase to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "vulnerability_addressed": self.vulnerability_addressed,
            "setup": self.setup,
            "test_code": self.test_code,
            "teardown": self.teardown,
            "expected_result": self.expected_result
        }


class TestCaseSet(BaseModel):
    test_cases: List[TestCase]

class TestCaseResponse(BaseModel):
    """API response model for test cases."""
    test_cases: List[TestCase] = Field(
        description="Generated test cases",
        min_items=1  # Ensure we have at least one test case
    )
    metrics: Dict[str, Any] = Field(
        description="Metrics about the generation process",
        default_factory=dict
    )

    @validator("test_cases", pre=True)
    def validate_test_cases(cls, v):
        """Ensure test cases are properly formatted."""
        if not v:
            raise ValueError("At least one test case is required")
        
        # Handle both list and dictionary inputs
        if isinstance(v, dict):
            v = list(v.values())
        
        # Convert each test case to proper format if needed
        formatted_cases = []
        for tc in v:
            if isinstance(tc, dict):
                formatted_cases.append(tc)
            elif hasattr(tc, "dict"):
                formatted_cases.append(tc.dict())
            else:
                raise ValueError(f"Invalid test case format: {type(tc)}")
                
        return formatted_cases

class GraphState(dict):
    """Represents the state of our graph."""
    attack_tree: str
    analysis: dict
    retrieval_prompt: str
    documents: List[Document]
    test_cases: List[dict]
    steps: List[str]
    document_grades: List[dict]
    search_needed: bool
    alignment_check: dict
    regeneration_attempts: int
    vulnerabilities: dict
    vuln_doc_pairs: List[dict] 
    vulnerability_coverage: dict 
    filtered_pairs: List[dict]