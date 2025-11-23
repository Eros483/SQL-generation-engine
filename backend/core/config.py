# ----- importable configurations @ src/core/config.py -----
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

load_dotenv()

class Settings(BaseSettings):
    """
    Central management for settings and configurations
    Reads .env file
    - Groq API Key
    - Persistent DB Path
    - MySQL database user
    - MySQL database passowrd
    - database host
    - database name
    """
    GROQ_API_KEY: str=os.getenv("GROQ_API_KEY", "")

    DB_USER: str=os.getenv("DB_USER", "root")
    DB_PASSWORD: str=os.getenv("DB_PASSWORD", "")
    DB_HOST: str=os.getenv("DB_HOST", "localhost")
    DB_NAME: str=os.getenv("DB_NAME", "fhs_coredb_local")

    Langsmith_API_KEY: str=os.getenv("LANGSMITH_API_KEY")
    GEMINI_API_KEY: str=os.getenv("GEMINI_API_KEY")
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()