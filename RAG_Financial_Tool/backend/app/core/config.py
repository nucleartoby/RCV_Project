from pydantic import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "RAG Financial Tool"

    CHROMA_PERSISTS_DIRECTORY: str = "./chroma_db"
    REDIS_URL: str = "redit://localhost:6379"
    
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-V2"
    FINANCIAL_EMBEDDING_MODEL: str = "ProsusAI/finbert"

    OPENAI_API_KEY: Optional[str] = None
    Antrhopics_API_KEY: Optional[str] = None

    SEC_USER_AGENT: str = "FinancialRAG toby.manwaring02@gmail.com"

    MAX_CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: str = 50

    CACHE_TTL: int = 3600

    class Config:
        env_file = ".env"

settings = Settings()