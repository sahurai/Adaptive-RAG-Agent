from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings loaded from the .env file.
    """
    GOOGLE_API_KEY: str
    GROQ_API_KEY: str
    TAVILY_API_KEY: str

    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_API_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()