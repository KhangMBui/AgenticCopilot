from fastapi import FastAPI
from app.routers.health import router as health_router
from app.routers.documents import router as documents_router
from app.routers.workspaces import router as workspaces_router
from app.routers.search import router as search_router
from app.routers.chat import router as chat_router

app = FastAPI(title="Agentic Copilot API")

app.include_router(health_router)
app.include_router(workspaces_router)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(chat_router)
