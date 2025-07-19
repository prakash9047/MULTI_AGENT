from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from the config directory
load_dotenv(dotenv_path="config/.env")
load_dotenv(dotenv_path="config/secrets.env")  # Also try secrets.env

class Settings(BaseSettings):
    # API Keys
    ALPHA_VANTAGE_API_KEY: Optional[str] = os.getenv("ALPHA_VANTAGE_API_KEY")
    FINNHUB_API_KEY: Optional[str] = os.getenv("FINNHUB_API_KEY")
    TWELVE_DATA_API_KEY: Optional[str] = os.getenv("TWELVE_DATA_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")  # Updated for Groq
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Vector Store
    FAISS_INDEX_PATH: str = "./data/vector_store"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM Settings - Updated for Groq with supported model
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3-8b-8192")  # Updated to supported model
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")  # Groq API
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.9"))
    
    # Voice Settings
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
    TTS_VOICE_ID: str = os.getenv("TTS_VOICE_ID", "default")
    
    # Performance
    MAX_CONCURRENT_AGENTS: int = int(os.getenv("MAX_CONCURRENT_AGENTS", "5"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "300"))
    
    # RAG Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    
    class Config:
        env_file = "config/secrets.env"
        extra = "ignore"  # Ignore extra fields

settings = Settings() 