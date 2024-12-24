from langchain_ollama import ChatOllama
from app.core.config import config
import logging
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
import json
import aiohttp
import yarl
import os
import asyncio

logger = logging.getLogger(__name__)

class LLMService:
    _instance = None

    def __init__(self):
        if LLMService._instance is not None:
            raise Exception("LLMService is a singleton class. Use get_instance() instead.")
        self.llm_config = config['llm']
        self.llm = self._init_llm(self.llm_config['model'])
        self.grading_llm = self._init_llm(self.llm_config['grader_model'])  # Separate instance for document grading
        self.json_parser = JsonOutputParser()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_llm(self, model_name):
        try:
            return ChatOllama(
                model=model_name,
                temperature=self.llm_config.get('temperature', 0),
                num_ctx=self.llm_config.get('max_tokens', 200000),
                format="json"
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM {model_name}: {e}")
            raise


    async def generate(self, prompt_dict, output_parser=None, use_grading_model=False):
        MAX_RETRIES = 3
        attempt = 0
        
        while attempt < MAX_RETRIES:
            try:
                # Convert dict to proper message format
                if isinstance(prompt_dict, dict) and "role" in prompt_dict and "content" in prompt_dict:
                    prompt = HumanMessage(content=prompt_dict["content"])
                else:
                    prompt = HumanMessage(content=str(prompt_dict))
                    
                # Use the appropriate model
                model = self.grading_llm if use_grading_model else self.llm
                
                # Add timeout based on observed successful times
                response = await asyncio.wait_for(
                    model.ainvoke([prompt]),
                    timeout=config["llm"]["timeout"]  # 3 minutes max based on successful runs
                )
                content = response.content
                # Use provided output parser or default JSON parser
                parser = output_parser or self.json_parser
                
                try:
                    if isinstance(content, str):
                        content = content.strip()
                        if content.startswith('```json'):
                            content = content.split('```json')[1]
                        if content.endswith('```'):
                            content = content.rsplit('```', 1)[0]
                        content = content.strip()
                    
                    if isinstance(parser, PydanticOutputParser):
                        return parser.parse(content)
                    else:
                        return parser.parse(content)
                        
                except Exception as parse_error:
                    logger.error(f"Failed to parse LLM response: {parse_error}")
                    logger.debug(f"Raw content: {content}")
                    return {
                        "error": "Failed to parse response",
                        "raw_content": content
                    }

            except asyncio.TimeoutError:
                attempt += 1
                logger.warning(f"Request timed out (attempt {attempt}/{MAX_RETRIES})")
                if attempt == MAX_RETRIES:
                    logger.error("Max retries reached, failing")
                    raise
                # Force restart Ollama on timeout
                os.system("docker restart docker_build-ollama-1")
                await asyncio.sleep(5)  # Wait for container to restart
                
            except Exception as e:
                attempt += 1
                logger.error(f"Request failed (attempt {attempt}/{MAX_RETRIES}): {str(e)}")
                if attempt == MAX_RETRIES:
                    raise
                await asyncio.sleep(2)  # Brief pause before retry

        raise Exception("Failed to get response after all retries")
