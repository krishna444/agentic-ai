
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API Keys
    GEMINI_API_KEY: str
    OPENAI_API_KEY: str
    GROQ_API_KEY: str
    
    # LLM Models
    GEMINI_MODEL: str= "gemma-4-31b-it"
    GMINI_PROVIDER:str= "google_genai"
    
    GROQ_MODEL:str="openai/gpt-oss-20b"
    GROQ_PROVIDER:str="openai"
    GROQ_BASE_URL:str='https://api.groq.com/openai/v1'
    
    GPT_MODEL:str="gpt-4o-mini"
    GPT_PROVIDER:str="openai"
    
    SYSTEM: str
    HUMAN: str
    
    model_config = SettingsConfigDict(env_file=".env",extra="ignore")

settings = Settings()