from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.base import Base  # noqa: F401 — re-exported for convenience
from app.settings import settings


engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
