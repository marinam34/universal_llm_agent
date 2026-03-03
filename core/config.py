from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "xiaomi/mimo-v2-flash"
    embedding_model: str = "openai/text-embedding-3-small"
    site_user_agent: str = "WebAgentGenerator/0.1"
    browser_headless: bool = True
    artifacts_dir: Path = Path("artifacts")
    frontend_origin: str = "http://127.0.0.1:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return settings
