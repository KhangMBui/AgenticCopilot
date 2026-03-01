from fastapi import FastAPI
from app.routers.health import router as health_router
from app.routers.documents import router as documents_router
from app.routers.workspaces import router as workspaces_router

app = FastAPI(title="Agentic Copilot API")

app.include_router(health_router)
app.include_router(workspaces_router)
app.include_router(documents_router)
