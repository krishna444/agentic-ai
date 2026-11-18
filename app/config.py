
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API Keys
    # LLM Models
    GOOGLE_API_KEY: str
    GOOGLE_MODEL: str
    GOOGLE_PROVIDER: str
    GOOGLE_BASE_URL: str
    
    GROQ_API_KEY: str
    GROQ_MODEL_OSS_120B:str
    GROQ_MODEL_OSS_20B:str
    GROQ_PROVIDER:str
    
    GPT_API_KEY:str
    GPT_MODEL:str
    GPT_PROVIDER:str
    GPT_BASE_URL:str
    
    OCI_LLM_API_KEY:str
    OCI_MODEL_GEMMA:str
    OCI_MODEL_QWEN:str
    OCI_LLM_BASE_URL:str

    OPENROUTER_API_KEY:str
    OPENROUTER_MODEL_DEEPSEEK:str
    OPENROUTER_MODEL_GLM:str
    OPENROUTER_BASE_URL:str
    OPENROUTER_PROVIDER:str

    SYSTEM: str
    HUMAN: str

    APP_USERS: str

    @property
    def users_map(self) -> dict[str, str]:
        users: dict[str, str] = {}
        for entry in self.APP_USERS.split(","):
            entry = entry.strip()
            if not entry:
                continue
            username, _, password = entry.partition(":")
            username, password = username.strip(), password.strip()
            if username and password:
                users[username] = password
        return users
    
    model_config = SettingsConfigDict(env_file=".env",extra="ignore")

settings = Settings()