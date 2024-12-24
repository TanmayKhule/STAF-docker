from pydantic_settings import BaseSettings
from typing import Optional
import yaml
from pathlib import Path

class Settings(BaseSettings):
    PROJECT_NAME: str = "STAF API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Load config from yaml
    @classmethod
    def load_config(cls):
        config_path = Path("config.yaml")
        if not config_path.exists():
            raise FileNotFoundError("config.yaml not found")
            
        with open(config_path) as f:
            return yaml.safe_load(f)

settings = Settings()
config = settings.load_config()
