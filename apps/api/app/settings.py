from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[3]  # project root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),  # looks for apps/api/.env
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str
    redis_host: str = "redis"
    redis_port: int = 6379


settings = Settings()
