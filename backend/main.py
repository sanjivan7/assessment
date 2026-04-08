from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import health, query, extract
from backend.config import settings

app = FastAPI(
    title="Khazanah Annual Review RAG API",
    description="Query and extract insights from Khazanah's Annual Review 2026 using AI.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS — allow Streamlit and Next.js frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router)
app.include_router(query.router)
app.include_router(extract.router)


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Khazanah Annual Review RAG API",
        "docs": "/docs",
        "health": "/health"
    }