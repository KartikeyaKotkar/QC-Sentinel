from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    CHROMA_PATH: str = "./data/chroma_db"
    CHROMA_COLLECTION: str = "qc_bugs"
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    TOP_K: int = 5
    DISTANCE_THRESHOLD: float = 0.35
    OLLAMA_HOST: str = "http://ollama:11434"
    LLM_PROVIDER: str = "ollama"
    LLM_MODEL: str = "ollama/llama3"
    LLM_TIMEOUT: int = 15
    LLM_MAX_RETRIES: int = 2
    API_KEY: str | None = None
    GITHUB_TOKEN: str | None = None


settings = Settings()
