from typing import Dict, List
from langgraph.graph import START, END, StateGraph
from app.models.schemas import GraphState, TestCase, TestCaseSet
from app.services.document_retrieval import DocumentRetriever
from app.services.llm_service import LLMService
from langchain_community.tools.tavily_search import TavilySearchResults
import json
import logging
import os
from datetime import datetime
from app.core.config import config
import time
import asyncio
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from app.models.schemas import DocumentRetrievalPrompt, AttackTreeAnalysisResult
from app.services.attack_tree import AttackTreeAnalyzer
from app.services.test_generation import TestCaseGenerator

logger = logging.getLogger(__name__)

class TestGenerationWorkflow:
    _instance_count = 0

    def _setup_logging(self):
        """Configure logging with detailed format"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def __init__(self):
        self._setup_logging()
        self.llm_service = LLMService.get_instance()
        self.attack_tree_analyzer = AttackTreeAnalyzer()
        self.document_retriever = DocumentRetriever()
        self.test_generator = TestCaseGenerator()
        # Use vectorstore directly like in the notebook
        self.vectorstore = self.document_retriever.vectorstore
        self.web_search_tool = TavilySearchResults()

        logger.info("TestGenerationWorkflow initialization complete")

    def _save_workflow_graph(self, workflow):
        """Save the workflow graph visualization as PNG"""
        try:
            # Create graphs directory if it doesn't exist
            graphs_dir = "graphs"
            os.makedirs(graphs_dir, exist_ok=True)

            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"workflow_graph_{timestamp}.png"
            filepath = os.path.join(graphs_dir, filename)

            # Get the graph visualization and save it
            graph_viz = workflow.get_graph()
            graph_png = graph_viz.draw_mermaid_png()
            
            # Save the PNG file
            with open(filepath, "wb") as f:
                f.write(graph_png)

            logger.info(f"Workflow graph saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save workflow graph: {e}")

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(GraphState)
        
        workflow.add_node("analyze_attack_tree", self._analyze_attack_tree)
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("generate_test_cases", self._generate_test_cases)
        workflow.add_node("check_test_cases", self._check_test_cases)
        workflow.add_node("regenerate_test_cases", self._regenerate_test_cases)

        # Build edges
        workflow.add_edge(START, "analyze_attack_tree")
        workflow.add_edge("analyze_attack_tree", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                "web_search": "web_search",
                "generate_test_cases": "generate_test_cases",
            },
        )
        workflow.add_edge("web_search", "generate_test_cases")
        workflow.add_edge("generate_test_cases", "check_test_cases")
        workflow.add_conditional_edges(
            "check_test_cases",
            self._decide_to_stop_or_regenerate,
            {
                END: END,
                "regenerate_test_cases": "regenerate_test_cases"
            }
        )
        workflow.add_edge("regenerate_test_cases", "check_test_cases")
        logger.info("Workflow graph building complete")

        return workflow.compile()

    async def _analyze_attack_tree(self, state: GraphState) -> GraphState:
        """Analyze attack tree using AttackTreeAnalyzer service"""
        start_time = time.time()

        try:
            attack_tree = json.loads(state["attack_tree"])
            
            # Use the AttackTreeAnalyzer service
            result = await self.attack_tree_analyzer.analyze(attack_tree)
            
            # Convert DocumentRetrievalPrompt to dictionary for state storage
            analysis = {
                "keywords": result.keywords,
                "vulnerabilities": result.vulnerabilities,
                "query": result.query
            }
            
            logger.info(f"Identified {len(result.vulnerabilities)} vulnerabilities")
            for vuln in result.vulnerabilities:
                logger.info(f"Vulnerability found: {vuln.get('name', 'Unknown')}")

            state.update({
                "analysis": analysis,
                "retrieval_prompt": result.query,
                "steps": state["steps"] + ["analyze_attack_tree"]
            })
            
            duration = time.time() - start_time
            logger.info(f"Attack tree analysis completed in {duration:.2f} seconds")
            return state
            
        except Exception as e:
            logger.error(f"Attack tree analysis failed: {e}")
            raise


    async def _retrieve_documents(self, state: GraphState) -> GraphState:
        """
        Retrieve documents based on the vulnerabilities.
        """
        start_time = time.time()
        try:
            vulnerabilities = state["analysis"]["vulnerabilities"]
            steps = state["steps"]
            steps.append("retrieve_documents")
            documents = []
            all_documents = []
            vuln_doc_pairs = []
            vulnerability_coverage = {vuln['name']: [] for vuln in vulnerabilities}


            for vulnerability in vulnerabilities:
                try:
                    query = f"{vulnerability['name']}: {vulnerability.get('description', '')}"
                    # Use similarity_search directly on vectorstore as in notebook
                    docs = self.vectorstore.similarity_search(query,k=config['vectorstore']['num_docs'])
                    for doc in docs:
                        if doc not in all_documents:
                            all_documents.append(doc)

                        vuln_doc_pairs.append({
                            "vulnerability": vulnerability['name'],
                            "document": doc,
                            "graded": False
                        })
                        vulnerability_coverage[vulnerability['name']].append(doc)

                    logger.info(f"Retrieved {len(docs)} documents for vulnerability {vulnerability['name']}")
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error retrieving documents for vulnerability: {str(e)}")
            
            state.update({
                        "documents": all_documents,  # Keep original documents list  # Add pairs
                        "vulnerability_coverage": vulnerability_coverage,
                        "vuln_doc_pairs":vuln_doc_pairs,
                        "steps": steps
                    })

            duration = time.time() - start_time
            logger.info(f"Document retrieval completed in {duration:.2f} seconds. Total documents: {len(documents)}")
            return state

            
        except Exception as e:
            logger.error(f"Document retrieval failed: {str(e)}")
            raise

    async def _grade_documents(self, state: GraphState) -> GraphState:
        """Grade retrieved documents for relevance"""
        start_time = time.time()
        logger.info("Starting document grading")
        try:
            vulnerabilities = state["analysis"]["vulnerabilities"]
            vuln_doc_pairs = state["vuln_doc_pairs"]
            steps = state["steps"]
            steps.append("grade_documents")
            filtered_pairs = []
            document_grades = []
            search_needed = False

            if not vuln_doc_pairs:
                logger.info("No documents retrieved, web search will be needed")
                state.update({
                    "documents": [],
                    "search_needed": True,
                    "steps": steps,
                    "document_grades": [],
                    "filtered_pairs": []
                })
                return state


            for pair in vuln_doc_pairs:
                if pair["graded"]:
                    continue
                    
                try:
                    prompt = {
                            "role": "user",
                            "content": f"""You are an expert security analyst grading the relevance of a retrieved document to vulnerabilities identified in an attack tree analysis.

    Vulnerability:
    {pair["vulnerability"]}

    Documents to grade:
    {pair["document"].page_content}

    Grade the document's relevance to the vulnerability based on the following criteria:
    1. The document discusses the specific vulnerability or closely related security issues.
    2. The document provides relevant information for understanding or mitigating the vulnerability.
    3. The document contains example code or test cases that could be adapted to test for this vulnerability.

    Provide a binary score as a JSON with a single key 'score':
    - Use 'yes' if the document is relevant and meets at least two of the above criteria.
    - Use 'no' if the document is not relevant or meets fewer than two criteria.

    Return only the JSON object with no preamble or explanation.
    Return only: {{"score": "yes"}} or {{"score": "no"}}"""
                    }
                    score = await self.llm_service.generate(prompt, use_grading_model=True)
                    grade = score.get("score", "no").lower()
                    
                    document_grades.append({
                        "vulnerability": pair["vulnerability"],
                        "document_content": pair["document"].page_content[:1000] + "...",
                        "grade": grade
                    })
                    pair["graded"] = True
                    if grade == "yes":
                        filtered_pairs.append(pair)

                except Exception as e:
                    logger.error(f"Error grading document for {pair['vulnerability']}: {str(e)}")
                    continue

            coverage = {}
            for pair in filtered_pairs:
                if pair["vulnerability"] not in coverage:
                    coverage[pair["vulnerability"]] = 0
                coverage[pair["vulnerability"]] += 1

            search_needed = any(count < 2 for count in coverage.values())
            if search_needed:
                logger.info("Insufficient coverage for some vulnerabilities, web search will be needed")
                for vuln, count in coverage.items():
                    if count < 2:
                        logger.info(f"Vulnerability {vuln} has only {count} relevant documents")

            # Get unique documents from filtered pairs
            unique_docs = []
            seen_contents = set()
            for pair in filtered_pairs:
                content = pair["document"].page_content
                if content not in seen_contents:
                    seen_contents.add(content)
                    unique_docs.append(pair["document"])

            state.update({
                "documents": unique_docs,
                "search_needed": search_needed,
                "steps": steps,
                "document_grades": document_grades,
                "filtered_pairs": filtered_pairs
            })

            duration = time.time() - start_time
            logger.info(f"Document grading completed in {duration:.2f} seconds")
            return state

        except Exception as e:
            logger.error(f"Document grading failed: {str(e)}")
            raise


    # async def _web_search(self, state: GraphState) -> GraphState:
    #     """Perform web search for vulnerabilities with insufficient coverage"""
    #     start_time = time.time()
    #     logger.info("Starting web search")
    #     try:
    #         steps = state["steps"]
    #         steps.append("web_search")
            
    #         # Get existing pairs and coverage
    #         existing_pairs = state.get("filtered_pairs", [])
    #         vuln_doc_pairs = state.get("vuln_doc_pairs", [])
            
    #         # Calculate current coverage
    #         coverage = {}
    #         for pair in existing_pairs:
    #             if pair["vulnerability"] not in coverage:
    #                 coverage[pair["vulnerability"]] = 0
    #             coverage[pair["vulnerability"]] += 1
            
    #         # Process vulnerabilities that need more documents
    #         new_pairs = []
    #         batch_size = 3  # Process vulnerabilities in batches
    #         vulnerabilities = state["analysis"]["vulnerabilities"]
    #         logger.info(f"Processing {len(vulnerabilities)} vulnerabilities for web search")
            
    #         for i in range(0, len(vulnerabilities), batch_size):
    #             batch = vulnerabilities[i:i + batch_size]
    #             logger.info(f"Processing batch {i//batch_size + 1}/{(len(vulnerabilities) + batch_size - 1)//batch_size}")
                
    #             for vulnerability in batch:
    #                 vuln_name = vulnerability.get('name', 'Unknown Vulnerability')
    #                 current_coverage = coverage.get(vuln_name, 0)
                    
    #                 if current_coverage >= 3:
    #                     logger.info(f"Skipping {vuln_name}: already has sufficient coverage")
    #                     continue
                    
    #                 try:
    #                     # Calculate how many more documents we need
    #                     docs_needed = 3 - current_coverage
    #                     description = vulnerability.get('description', '')
    #                     query = f"{vuln_name} - {description} security vulnerability test cases"
    #                     logger.info(f"Performing web search for: {vuln_name} (need {docs_needed} more documents)")
                        
    #                     web_results = await self.web_search_tool.ainvoke({
    #                         "query": query,
    #                         "max_results": docs_needed
    #                     })
                        
    #                     # Create new vulnerability-document pairs
    #                     for result in web_results:
    #                         new_pair = {
    #                             "vulnerability": vuln_name,
    #                             "document": Document(
    #                                 page_content=result["content"],
    #                                 metadata={"url": result["url"]}
    #                             ),
    #                             "graded": False,
    #                             "source": "web_search"
    #                         }
    #                         new_pairs.append(new_pair)
                            
    #                     logger.info(f"Retrieved {len(web_results)} documents from web search for {vuln_name}")
                        
    #                 except Exception as e:
    #                     logger.error(f"Error in web search for {vuln_name}: {str(e)}")
    #                     continue
            
    #         # Combine existing and new pairs
    #         combined_pairs = vuln_doc_pairs + new_pairs
            
    #         # Update state with new pairs
    #         state.update({
    #             "vuln_doc_pairs": combined_pairs,
    #             "steps": steps
    #         })
            
    #         duration = time.time() - start_time
    #         logger.info(f"Web search completed in {duration:.2f} seconds. Added {len(new_pairs)} new pairs")
            
    #         # Log coverage summary
    #         logger.info("Coverage summary after web search:")
    #         final_coverage = {}
    #         for pair in combined_pairs:
    #             if pair["vulnerability"] not in final_coverage:
    #                 final_coverage[pair["vulnerability"]] = 0
    #             final_coverage[pair["vulnerability"]] += 1
                
    #         for vuln, count in final_coverage.items():
    #             logger.info(f"- {vuln}: {count} documents")
            
    #         return state

    #     except Exception as e:
    #         logger.error(f"Web search failed: {str(e)}")
    #         raise


    async def _web_search(self, state: GraphState) -> GraphState:
        """Perform web search for vulnerabilities with insufficient coverage"""
        start_time = time.time()
        logger.info("Starting web search")
        try:
            steps = state["steps"]
            steps.append("web_search")
            
            existing_pairs = state.get("filtered_pairs", [])
            vuln_doc_pairs = state.get("vuln_doc_pairs", [])
            
            # Calculate current coverage
            coverage = {}
            for pair in existing_pairs:
                if pair["vulnerability"] not in coverage:
                    coverage[pair["vulnerability"]] = 0
                coverage[pair["vulnerability"]] += 1
            
            # Process vulnerabilities that need more documents
            new_pairs = []
            batch_size = 3
            vulnerabilities = state["analysis"]["vulnerabilities"]
            logger.info(f"Processing {len(vulnerabilities)} vulnerabilities for web search")
            
            for i in range(0, len(vulnerabilities), batch_size):
                batch = vulnerabilities[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(vulnerabilities) + batch_size - 1)//batch_size}")
                
                for vulnerability in batch:
                    vuln_name = vulnerability.get('name', 'Unknown Vulnerability')
                    current_coverage = coverage.get(vuln_name, 0)
                    
                    if current_coverage >= 3:
                        logger.info(f"Skipping {vuln_name}: already has sufficient coverage")
                        continue
                    
                    try:
                        docs_needed = 3 - current_coverage
                        description = vulnerability.get('description', '')
                        query = f"{vuln_name} - {description} security vulnerability test cases"
                        logger.info(f"Performing web search for: {vuln_name} (need {docs_needed} more documents)")
                        
                        # Invoke Tavily search
                        raw_results = await self.web_search_tool.ainvoke({
                            "query": query,
                            "max_results": docs_needed
                        })
                        
                        # Tavily returns a list of search results
                        if isinstance(raw_results, list):
                            for result in raw_results:
                                # Create content from available fields
                                content = ""
                                if isinstance(result, dict):
                                    content = f"Title: {result.get('title', '')}\n\n"
                                    content += f"Content: {result.get('content', '')}\n\n"
                                    content += f"Snippet: {result.get('snippet', '')}"
                                    url = result.get('url', '')
                                else:
                                    # If result is a string, use it directly
                                    content = str(result)
                                    url = "unknown"
                                
                                new_pair = {
                                    "vulnerability": vuln_name,
                                    "document": Document(
                                        page_content=content,
                                        metadata={"url": url, "source": "web_search"}
                                    ),
                                    "graded": False,
                                    "source": "web_search"
                                }
                                new_pairs.append(new_pair)
                                
                            logger.info(f"Retrieved {len(raw_results)} documents from web search for {vuln_name}")
                            
                    except Exception as e:
                        logger.error(f"Error in web search for {vuln_name}: {str(e)}")
                        logger.debug(f"Raw results: {raw_results if 'raw_results' in locals() else 'No results'}")
                        continue
            
            # Combine existing and new pairs
            combined_pairs = vuln_doc_pairs + new_pairs
            
            # Update state with new pairs
            state.update({
                "vuln_doc_pairs": combined_pairs,
                "steps": steps
            })
            
            duration = time.time() - start_time
            logger.info(f"Web search completed in {duration:.2f} seconds. Added {len(new_pairs)} new pairs")
            
            # Log coverage summary
            logger.info("Coverage summary after web search:")
            final_coverage = {}
            for pair in combined_pairs:
                if pair["vulnerability"] not in final_coverage:
                    final_coverage[pair["vulnerability"]] = 0
                final_coverage[pair["vulnerability"]] += 1
                
            for vuln, count in final_coverage.items():
                logger.info(f"- {vuln}: {count} documents")
            
            return state

        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            logger.exception("Detailed traceback:")
            raise



    async def _generate_test_cases(self, state: GraphState) -> GraphState:
        """Generate test cases maintaining dictionary structure"""
        start_time = time.time()
        logger.info("Starting test case generation")
        
        try:
            test_cases = await self.test_generator.generate_test_cases(
                json.loads(state["attack_tree"]),
                state["analysis"],
                state["documents"]
            )
            
            # Verify we have a dictionary
            if not isinstance(test_cases, dict):
                raise ValueError("Test cases must be a dictionary")
                
            # Log test case information
            logger.info(f"Generated {len(test_cases)} test cases")
            for tc_id, tc in test_cases.items():
                logger.info(f"Test case {tc_id}: {tc.name}")
            
            state.update({
                "test_cases": test_cases,
                "steps": state["steps"] + ["generate_test_cases"]
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Test case generation failed: {str(e)}")
            raise



    def _assign_unique_numbers(self, test_cases):
        """Assign unique numbers to test cases handling both dict and Pydantic objects"""
        return [
            {
                "id": i + 1,
                "name": f"Test Case {i + 1}: {tc.get('name') if isinstance(tc, dict) else tc.name}",
                "description": tc.get('description') if isinstance(tc, dict) else tc.description,
                "vulnerability_addressed": tc.get('vulnerability_addressed') if isinstance(tc, dict) else tc.vulnerability_addressed,
                "setup": tc.get('setup') if isinstance(tc, dict) else tc.setup,
                "test_code": tc.get('test_code') if isinstance(tc, dict) else tc.test_code,
                "teardown": tc.get('teardown') if isinstance(tc, dict) else tc.teardown,
                "expected_result": tc.get('expected_result') if isinstance(tc, dict) else tc.expected_result
            }
            for i, tc in enumerate(test_cases)
        ]


    def _format_documents(self, documents: List[Document]) -> str:
        return "\n\n".join(
            f"Document {i+1}:\n{doc.page_content}" 
            for i, doc in enumerate(documents)
        )

    async def _check_test_cases(self, state: GraphState) -> GraphState:
        """Check test cases with enhanced logging."""
        logger.info("=== ENTERING TEST CASE CHECK PHASE ===")
        try:
            start_time = time.time()
            test_cases = state["test_cases"]
            logger.info(f"State contains {len(test_cases)} test cases to check")
            
            attack_tree = json.loads(state["attack_tree"])
            logger.info("Successfully parsed attack tree for checking")
            
            logger.info("Preparing to send check request to LLM service...")
            response_start = time.time()
            alignment_check = await self.test_generator.check_test_cases(
                attack_tree,
                test_cases
            )
            logger.info(f"LLM check request completed in {time.time() - response_start:.2f} seconds")
            
            if alignment_check:
                logger.info(f"Received alignment check results:")
                logger.info(f"- Alignment Score: {alignment_check.get('alignment_score')}")
                logger.info(f"- Missing Vulnerabilities: {len(alignment_check.get('missing_vulnerabilities', []))}")
                logger.info(f"- Improvement Suggestions: {len(alignment_check.get('improvement_suggestions', []))}")
            else:
                logger.warning("Received empty alignment check results")
                
            state.update({
                "alignment_check": alignment_check,
                "steps": state["steps"] + ["check_test_cases"]
            })
            logger.info(f"Test case check phase completed in {time.time() - start_time:.2f} seconds")
            return state
            
        except Exception as e:
            logger.error(f"=== TEST CASE CHECK FAILED ===")
            logger.error(f"Error details: {str(e)}")
            logger.exception("Full traceback:")
            raise

    async def _regenerate_test_cases(self, state: GraphState) -> GraphState:
        """Handle test case regeneration with proper error handling."""
        try:
            regeneration_attempts = state.get("regeneration_attempts", 0) + 1
            logger.info(f"=== STARTING REGENERATION ATTEMPT {regeneration_attempts} ===")

            if regeneration_attempts >= config['llm']['max_attempts']:
                logger.info("Maximum regeneration attempts reached, keeping best test cases")
                return {**state, "regeneration_attempts": regeneration_attempts}

            start_time = time.time()
            test_cases = state["test_cases"]
            alignment_check = state["alignment_check"]

            improved_test_cases = await self.test_generator.regenerate_test_cases(
                test_cases,
                alignment_check.get("missing_vulnerabilities", []),
                alignment_check.get("improvement_suggestions", [])
            )

            duration = time.time() - start_time
            logger.info(f"Regeneration attempt completed in {duration:.2f} seconds")

            # Update state with new test cases
            return {
                **state,
                "test_cases": improved_test_cases,
                "regeneration_attempts": regeneration_attempts,
                "steps": state["steps"] + ["regenerate_test_cases"]
            }

        except Exception as e:
            logger.error(f"Regeneration attempt failed: {e}")
            # Return current state with incremented attempt counter
            return {**state, "regeneration_attempts": regeneration_attempts}



    def _decide_to_generate(self, state: GraphState) -> str:
        """Decide whether to generate test cases or perform web search."""
        next_step = "web_search" if state.get("search_needed", False) else "generate_test_cases"
        logger.info(f"Workflow decision: proceeding to {next_step}")
        return next_step


    def _decide_to_stop_or_regenerate(self, state: GraphState) -> str:
        """Make the final decision whether to stop or continue regeneration."""
        logger.info("=== EVALUATING WORKFLOW DECISION POINT ===")
        
        regeneration_attempts = state.get("regeneration_attempts", 0)
        alignment_check = state.get("alignment_check", {})
        
        # Get scores
        alignment_score = alignment_check.get("alignment_score", 0)
        completeness_score = alignment_check.get("completeness_score", 0)
        quality_score = alignment_check.get("quality_score", 0)
        
        logger.info(f"Current regeneration attempt: {regeneration_attempts}")
        logger.info(f"Quality scores - Alignment: {alignment_score}, "
                f"Completeness: {completeness_score}, Quality: {quality_score}")

        # Check stop conditions
        if regeneration_attempts >= config['llm']['max_attempts']:
            logger.info("Maximum regeneration attempts reached")
            # Format final output
            self._format_final_output(state)
            return END
            
        if (alignment_score >= 90 and 
            completeness_score >= 90 and 
            quality_score >= 90):
            logger.info("Quality thresholds met")
            # Format final output
            self._format_final_output(state)
            return END
            
        # Continue regeneration
        logger.info("Continuing with test case regeneration")
        return "regenerate_test_cases"


    async def run(self, attack_tree: Dict) -> Dict:
        """
        Execute the complete workflow and ensure proper test case output.
        Returns a dictionary containing test cases and metrics.
        """
        total_start_time = time.time()
        logger.info("Starting complete test generation workflow")

        try:
            # Build and run workflow
            workflow = self._build_workflow()
            result = await workflow.ainvoke({
                "attack_tree": json.dumps(attack_tree),
                "steps": []
            })

            # Extract test cases from state
            test_cases = result.get("final_test_cases", [])
            metrics = result.get("final_metrics", {})

            logger.info(f"Workflow extracted {len(test_cases)} test cases")

            # Validate we have test cases
            if not test_cases:
                logger.warning("No test cases in final output, checking state")
                # Fallback to test_cases in state if final_test_cases is empty
                state_test_cases = result.get("test_cases", {})
                if isinstance(state_test_cases, dict):
                    # Convert dictionary to list
                    test_cases = [tc.dict() if hasattr(tc, 'dict') else tc 
                                for tc in state_test_cases.values()]
                    logger.info(f"Retrieved {len(test_cases)} test cases from state")
                else:
                    test_cases = []

            # Ensure test cases are properly formatted
            formatted_test_cases = []
            for tc in test_cases:
                if isinstance(tc, dict):
                    formatted_test_cases.append(tc)
                elif hasattr(tc, 'dict'):
                    formatted_test_cases.append(tc.dict())

            logger.info(f"Successfully formatted {len(formatted_test_cases)} test cases")
            
            return {
                "test_cases": formatted_test_cases,
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            logger.exception("Full traceback:")
            raise

    def _verify_test_cases(self, test_cases):
        """Verify test cases and remove duplicates"""
        seen_ids = set()
        verified_test_cases = []
        
        for tc in test_cases:
            if tc["id"] not in seen_ids:
                seen_ids.add(tc["id"])
                verified_test_cases.append(tc)
                
        return verified_test_cases

    print("=== LOADING WORKFLOW MODULE ===")

    def _format_final_output(self, state: GraphState) -> None:
        """Format final test cases and metrics before ending."""
        try:
            logger.info("Formatting final output...")
            test_cases = state.get("test_cases", {})
            alignment_check = state.get("alignment_check", {})
            
            # Format test cases
            formatted_cases = []
            for tc_id, tc in test_cases.items():
                try:
                    if isinstance(tc, TestCase):
                        tc_dict = tc.dict()
                    else:
                        tc_dict = tc.copy()
                    formatted_cases.append(tc_dict)
                    logger.info(f"Formatted test case {tc_id}")
                except Exception as e:
                    logger.error(f"Error formatting test case {tc_id}: {e}")
                    continue
            
            # Sort by ID
            formatted_cases.sort(key=lambda x: x["id"])
            
            # Store final results
            state["final_test_cases"] = formatted_cases
            state["final_metrics"] = {
                "alignment_score": alignment_check.get("alignment_score", 0),
                "completeness_score": alignment_check.get("completeness_score", 0),
                "quality_score": alignment_check.get("quality_score", 0),
                "regeneration_attempts": state.get("regeneration_attempts", 0)
            }
            
            logger.info("Final output formatting complete")
            
        except Exception as e:
            logger.error(f"Error formatting final output: {e}")
            # Ensure we have something in the state
            state["final_test_cases"] = list(state.get("test_cases", {}).values())
            state["final_metrics"] = {}