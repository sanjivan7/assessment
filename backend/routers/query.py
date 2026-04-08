from fastapi import APIRouter, HTTPException
from openai import OpenAI

from backend.models.schemas import QueryRequest, QueryResponse, ConfidenceLevel
from backend.services.retriever import retrieve_chunks, load_full_context
from backend.services.llm import get_rag_answer, get_full_context_answer
from backend.config import settings

router = APIRouter()


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_document(request: QueryRequest):
    """
    Query the Annual Review using RAG or full-context mode.

    - RAG mode (default): retrieves top-k relevant slides, answers from those
    - Full-context mode: sends entire document to LLM (more accurate, higher cost)
    """
    client = get_openai_client()

    try:
        if request.full_context_mode:
            # Full context mode — bypass retrieval
            full_context = load_full_context()
            answer = get_full_context_answer(request.query, full_context, client)
            return QueryResponse(
                answer=answer,
                sources=[],
                confidence=ConfidenceLevel.HIGH,
                retrieval_scores=[],
                full_context_mode_used=True,
                query=request.query
            )

        # RAG mode — retrieve then generate
        sources, scores, confidence = retrieve_chunks(
            query=request.query,
            client=client,
            top_k=request.top_k
        )

        answer = get_rag_answer(
            query=request.query,
            sources=sources,
            confidence=confidence,
            client=client
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            retrieval_scores=scores,
            full_context_mode_used=False,
            query=request.query
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Document not ingested yet: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )