"""
Retrieval service — queries ChromaDB and returns ranked chunks.
Also supports full-context mode where all slides are loaded directly.
"""

import json
from pathlib import Path
from openai import OpenAI
import chromadb

from backend.config import settings
from backend.models.schemas import SourceChunk, ConfidenceLevel


def get_chroma_collection() -> chromadb.Collection:
    """Return the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_collection(settings.collection_name)


def embed_query(query: str, client: OpenAI) -> list[float]:
    """Embed a user query using the same model used during ingestion."""
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=[query]
    )
    return response.data[0].embedding


def retrieve_chunks(
    query: str,
    client: OpenAI,
    top_k: int = 5
) -> tuple[list[SourceChunk], list[float], ConfidenceLevel]:
    """
    Retrieve top-k most relevant chunks for a query.

    Returns:
        sources: list of SourceChunk with metadata
        scores: list of relevance scores
        confidence: HIGH / MEDIUM / LOW / NO_CONTEXT
    """
    collection = get_chroma_collection()

    # Embed the query
    query_embedding = embed_query(query, client)

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    # ChromaDB returns distances (lower = more similar for cosine)
    # Convert to similarity scores (higher = more similar)
    distances = results["distances"][0]
    scores = [round(1 - d, 4) for d in distances]

    # Build SourceChunk objects
    sources = []
    for i, (doc, meta, score) in enumerate(zip(documents, metadatas, scores)):
        sources.append(SourceChunk(
            chunk_id=f"slide_{meta['page_number']:03d}",
            page_number=meta["page_number"],
            slide_title=meta.get("slide_title", f"Slide {meta['page_number']}"),
            slide_type=meta.get("slide_type", "unknown"),
            extraction_method=meta.get("extraction_method", "unknown"),
            relevance_score=score,
            text_preview=doc[:200] + "..." if len(doc) > 200 else doc
        ))

    # Determine confidence from top score
    top_score = scores[0] if scores else 0.0
    if top_score >= 0.70:
        confidence = ConfidenceLevel.HIGH
    elif top_score >= 0.45:
        confidence = ConfidenceLevel.MEDIUM
    elif top_score >= settings.similarity_threshold:
        confidence = ConfidenceLevel.LOW
    else:
        confidence = ConfidenceLevel.NO_CONTEXT

    return sources, scores, confidence


def load_full_context() -> str:
    """Load the full document context for full-context mode."""
    full_context_path = settings.extracted_dir / "full_context.txt"
    if not full_context_path.exists():
        raise FileNotFoundError(
            "full_context.txt not found. Run scripts/ingest.py first."
        )
    return full_context_path.read_text(encoding="utf-8")


def load_extracted_data() -> dict:
    """Load the structured extracted data for the /extract endpoint."""
    extracted_path = settings.extracted_dir / "structured_data.json"
    if not extracted_path.exists():
        raise FileNotFoundError(
            "structured_data.json not found. Run scripts/ingest.py first."
        )
    with open(extracted_path, "r", encoding="utf-8") as f:
        return json.load(f)