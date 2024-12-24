
from pydantic_settings import BaseSettings
from typing import Optional
import yaml
from pathlib import Path
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "STAF API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Use only the environment variables present in your original code
    HF_TOKEN: Optional[str] = None
    HUGGINGFACE_TOKEN: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
    
    class Config:
        env_file = ".env"

    def load_config(self) -> dict:
        """
        Load configuration with the following precedence:
        1. Environment variables (only those in original code)
        2. Mounted config.yaml
        3. Default config.yaml
        """
        # Start with empty config
        config_dict = {}
        
        # Load default config if it exists
        default_config_path = Path("/config/default_config.yaml")
        if default_config_path.exists():
            with open(default_config_path) as f:
                config_dict = yaml.safe_load(f)
        
        # Load custom config if it exists
        config_path = Path("config.yaml")
        if config_path.exists():
            with open(config_path) as f:
                custom_config = yaml.safe_load(f)
                if custom_config:
                    self._merge_configs(config_dict, custom_config)
        
        # Override with environment variables if they're set
        if self.HF_TOKEN:
            config_dict.setdefault("huggingface", {})["api_token"] = self.HF_TOKEN
        if self.HUGGINGFACE_TOKEN:
            config_dict.setdefault("huggingface", {})["api_token"] = self.HUGGINGFACE_TOKEN
        if self.TAVILY_API_KEY:
            config_dict.setdefault("tavily", {})["api_key"] = self.TAVILY_API_KEY

        return config_dict
    
    def _merge_configs(self, base_config: dict, override_config: dict):
        """Deep merge override_config into base_config"""
        for key, value in override_config.items():
            if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
                self._merge_configs(base_config[key], value)
            else:
                base_config[key] = value

settings = Settings()
config = settings.load_config()
