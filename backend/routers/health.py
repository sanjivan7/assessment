from fastapi import APIRouter
from backend.models.schemas import HealthResponse
from backend.config import settings
from pathlib import Path

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system status and ingestion state."""
    chunk_count = 0
    pdf_ingested = False

    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        collection = client.get_collection(settings.collection_name)
        chunk_count = collection.count()
        pdf_ingested = chunk_count > 0
    except Exception:
        pass

    status = "healthy" if pdf_ingested else "degraded"
    message = (
        f"System operational. {chunk_count} chunks loaded."
        if pdf_ingested
        else "PDF not yet ingested. Run scripts/ingest.py first."
    )

    return HealthResponse(
        status=status,
        chunk_count=chunk_count,
        collection_name=settings.collection_name,
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        pdf_ingested=pdf_ingested,
        message=message
    )