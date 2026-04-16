"""
PURPOSE:
Retrieval service responsible for interacting with ChromaDB and the file system.
It converts natural language queries into embeddings, performs similarity searches,
and handles full-document context loading for the Khazanah Annual Review 2026.
"""

import json
from pathlib import Path
from openai import OpenAI
import chromadb

from backend.config import settings
from backend.models.schemas import SourceChunk, ConfidenceLevel


def get_chroma_collection() -> chromadb.Collection:
    """
    Initializes a persistent ChromaDB client and retrieves the specific 
    collection defined in the system settings.
    """
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_collection(settings.collection_name)


def embed_query(query: str, client: OpenAI) -> list[float]:
    """
    Converts a plain text user query into a high-dimensional vector using 
    OpenAI's embedding model. This vector is used for mathematical 
    similarity comparison.
    """
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
    The core RAG retrieval logic:
    1. Embeds the user query.
    2. Searches ChromaDB for the most relevant text segments.
    3. Converts distance metrics into similarity scores.
    4. Evaluates system confidence based on retrieval quality.

    Returns:
        sources: list of SourceChunk objects containing text and metadata.
        scores: list of raw similarity scores (0.0 to 1.0).
        confidence: An Enum indicating the reliability of the retrieved context.
    """
    collection = get_chroma_collection()

    # Step 1: Generate the embedding for the incoming query
    query_embedding = embed_query(query, client)

    # Step 2: Query the vector database
    # include=["documents", "metadatas", "distances"] ensures we get all needed info
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    
    # Step 3: Math Conversion
    # ChromaDB returns distances (where lower is better).
    # We convert to similarity scores (where 1.0 is a perfect match) for better readability.
    distances = results["distances"][0]
    scores = [round(1 - d, 4) for d in distances]

    # Step 4: Map raw results to the SourceChunk Pydantic model
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

    # Step 5: Confidence Scoring Logic
    # We use the highest similarity score to determine how 'sure' we are.
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
    """
    Reads the entire pre-processed document text from disk.
    Used for 'Full Context Mode' when the LLM needs to analyze the 
    whole report instead of specific chunks.
    """
    full_context_path = settings.extracted_dir / "full_context.txt"
    if not full_context_path.exists():
        raise FileNotFoundError(
            "full_context.txt not found. Run scripts/ingest.py first."
        )
    return full_context_path.read_text(encoding="utf-8")


def load_extracted_data() -> dict:
    """
    Loads the structured JSON data (financial metrics, portfolio tables) 
    that was extracted during the ingestion phase.
    """
    extracted_path = settings.extracted_dir / "structured_data.json"
    if not extracted_path.exists():
        raise FileNotFoundError(
            "structured_data.json not found. Run scripts/ingest.py first."
        )
    with open(extracted_path, "r", encoding="utf-8") as f:
        return json.load(f)
