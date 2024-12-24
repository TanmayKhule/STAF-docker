from fastapi import FastAPI, HTTPException
from app.models.schemas import AttackTreeRequest, TestCaseResponse
from app.services.workflow import TestGenerationWorkflow
from app.core.config import config
import asyncio
import logging
from pydantic import BaseModel


logger = logging.getLogger(__name__)

print("=== STARTING MAIN MODULE LOAD ===")
app = FastAPI(title="STAF API")

async def pull_ollama_models():
    """Pull required Ollama models at startup"""
    models_to_pull = config['llm'].get('models_to_pull', [])
    
    for model in models_to_pull:
        try:
            logger.info(f"Pulling model: {model}")
            process = await asyncio.create_subprocess_exec(
                'ollama', 'pull', model,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Successfully pulled model: {model}")
            else:
                logger.error(f"Error pulling model {model}: {stderr.decode()}")
        except Exception as e:
            logger.error(f"Exception while pulling model {model}: {str(e)}")

@app.on_event("startup")
async def startup_event():
    print("=== FASTAPI STARTUP EVENT ===")
    await pull_ollama_models()

print("=== CREATING WORKFLOW INSTANCE ===")


workflow = TestGenerationWorkflow()

print(f"Created workflow instance at {id(workflow)}")

# @app.post("/api/v1/generate-tests", response_model=TestCaseResponse)
# async def generate_test_cases(request: AttackTreeRequest):
#     try:
#         result = await workflow.run(request.attack_tree.dict())
        
#         if not result.get("test_cases"):
#             raise HTTPException(
#                 status_code=500,
#                 detail="No test cases were generated. Please try again."
#             )
            
#         return TestCaseResponse(
#             test_cases=result["test_cases"],
#             metrics=result.get("metrics", {})
#         )
#     except Exception as e:
#         logger.error(f"Error generating test cases: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=str(e)
#         )

@app.post("/api/v1/generate-tests", response_model=TestCaseResponse)
async def generate_test_cases(request: AttackTreeRequest):
    """Generate security test cases from an attack tree."""
    try:
        logger.info("Starting test case generation")
        
        # Run the workflow
        result = await workflow.run(request.attack_tree.dict())
        
        # Extract and validate test cases
        test_cases = result.get("test_cases", [])
        metrics = result.get("metrics", {})
        
        logger.info(f"Retrieved {len(test_cases)} test cases from workflow")
        
        if not test_cases:
            logger.error("No test cases found in workflow result")
            raise HTTPException(
                status_code=500,
                detail="No test cases were generated. Please try again."
            )

        # Create response with the test cases and metrics
        response = TestCaseResponse(
            test_cases=test_cases,
            metrics=metrics
        )
        
        logger.info(f"Successfully created response with {len(test_cases)} test cases")
        return response
        
    except Exception as e:
        logger.error(f"Error generating test cases: {e}")
        logger.exception("Detailed traceback:")
        
        if isinstance(e, HTTPException):
            raise
            
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

class VectorstoreQuery(BaseModel):
    query: str
    k: int = config["vectorstore"]["num_docs"]


@app.post("/api/v1/query-vectorstore")
async def query_vectorstore(query_input: VectorstoreQuery):
    try:
        from app.services.document_retrieval import DocumentRetriever
        retriever = DocumentRetriever()
        results = retriever.query_vectorstore(query_input.query, query_input.k)
        return {
            "results": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}