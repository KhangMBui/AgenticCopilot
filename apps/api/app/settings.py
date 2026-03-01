from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Walk up from this file to find a .env (local dev).
# In Docker, env vars are injected directly so no file is needed.
def _find_env_file() -> str | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate.exists():
            return str(candidate)
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str
    redis_host: str = "redis"
    redis_port: int = 6379

    # OpenAI API key for embeddings
    openai_api_key: str


settings = Settings()
