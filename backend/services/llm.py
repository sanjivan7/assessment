"""
LLM service — handles prompt construction and GPT-4o-mini calls.
"""

from openai import OpenAI
from backend.models.schemas import SourceChunk, ConfidenceLevel
from backend.config import settings


RAG_SYSTEM_PROMPT = """You are an AI analyst assistant for Khazanah Nasional Berhad's Annual Review 2026.

STRICT RULES — follow these without exception:
1. Answer ONLY using the provided context slides. Never use outside knowledge.
2. If the answer is not in the context, respond with exactly:
   "This information is not available in the provided Annual Review document."
3. Always include units for financial figures (RM billions, %, x for coverage ratios).
4. Always specify the year when citing financial data.
5. End every answer with: "Sources: Slide [X], Slide [Y]" listing the slides you used.
6. If a chart is described but exact values are uncertain, say "approximately" before the value.
7. Never fabricate numbers. If unsure, say so explicitly.
8. Keep answers concise and factual — this is for professional analyst use."""


FULL_CONTEXT_SYSTEM_PROMPT = """You are an AI analyst assistant for Khazanah Nasional Berhad's Annual Review 2026.
You have been provided with the complete extracted content of the Annual Review.

STRICT RULES:
1. Answer ONLY using the provided document content.
2. If the answer is not present, say: "This information is not available in the provided Annual Review document."
3. Always include units (RM billions, %, x) and years for financial data.
4. Reference specific slide numbers when citing information.
5. Never fabricate numbers.
6. Keep answers concise and factual."""


def build_rag_context(sources: list[SourceChunk], documents: list[str]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    context_parts = []
    for i, (source, doc_text) in enumerate(zip(sources, documents)):
        context_parts.append(
            f"--- SLIDE {source.page_number}: {source.slide_title} "
            f"[Relevance: {source.relevance_score:.2f}] ---\n{doc_text}"
        )
    return "\n\n".join(context_parts)


def get_rag_answer(
    query: str,
    sources: list[SourceChunk],
    confidence: ConfidenceLevel,
    client: OpenAI,
    collection=None
) -> str:
    """
    Generate an answer using retrieved chunks as context.
    If confidence is NO_CONTEXT, return early without LLM call.
    """
    # Gate: if no relevant context found, don't call LLM
    if confidence == ConfidenceLevel.NO_CONTEXT:
        return (
            "This information is not available in the provided Annual Review document. "
            "The query did not match any content in the document with sufficient confidence."
        )

    # Fetch full document texts for the retrieved chunks
    if collection is None:
        from backend.services.retriever import get_chroma_collection
        collection = get_chroma_collection()

    chunk_ids = [s.chunk_id for s in sources]
    results = collection.get(ids=chunk_ids, include=["documents"])
    documents = results["documents"]

    context = build_rag_context(sources, documents)

    user_message = f"""CONTEXT FROM ANNUAL REVIEW:
{context}

QUESTION: {query}

Answer based strictly on the context above."""

    response = client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=800,
        temperature=0.1,  # Low temperature for factual accuracy
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )

    return response.choices[0].message.content.strip()


def get_full_context_answer(
    query: str,
    full_context: str,
    client: OpenAI
) -> str:
    """
    Generate an answer by stuffing the full document into context.
    Fallback mode — bypasses retrieval entirely.
    """
    user_message = f"""COMPLETE ANNUAL REVIEW CONTENT:
{full_context}

QUESTION: {query}

Answer based strictly on the document content above."""

    response = client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=800,
        temperature=0.1,
        messages=[
            {"role": "system", "content": FULL_CONTEXT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )

    return response.choices[0].message.content.strip()