"""
PURPOSE:
This module defines the core data structures and validation schemas for the 
Khazanah Annual Review 2026 AI system. It handles:
1. User query parameters (RAG configuration).
2. Structured extraction of financial metrics and portfolio data from PDFs.
3. Standardized API responses for retrieved chunks and LLM-generated answers.
4. System health monitoring and state tracking.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

# --- ENUMERATIONS ---
class ConfidenceLevel(str, Enum):
    """Enumeration for the AI's self-assessed certainty in its answer."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NO_CONTEXT = "no_context"

# --- RAG PIPELINE MODELS ---
class QueryRequest(BaseModel):
    """Schema for incoming user questions and retrieval settings."""
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language question about the Annual Review"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve"
    )
    full_context_mode: bool = Field(
        default=False,
        description="If True, send entire document to LLM instead of retrieved chunks"
    )

class SourceChunk(BaseModel):
    """Metadata and content for a single retrieved piece of document text."""
    chunk_id: str
    page_number: int
    slide_title: str
    slide_type: str
    extraction_method: str
    relevance_score: float
    text_preview: str  # first 200 chars of chunk text

class QueryResponse(BaseModel):
    """The final payload returned to the user, combining the answer and its citations."""
    answer: str
    sources: list[SourceChunk]
    confidence: ConfidenceLevel
    retrieval_scores: list[float]
    full_context_mode_used: bool = False
    query: str

# --- STRUCTURED DATA EXTRACTION MODELS ---
class ExtractedMetric(BaseModel):
    """Schema for financial data points found within the text or tables."""
    metric_name: str
    value: str
    unit: Optional[str] = None
    year: Optional[str] = None
    context: Optional[str] = None

class PortfolioCompany(BaseModel):
    """Details regarding specific companies mentioned in the report."""
    name: str
    sector: Optional[str] = None
    ownership_stake: Optional[str] = None
    notes: Optional[str] = None

class ExtractedData(BaseModel):
    """The primary container for all structured entities pulled from the PDF."""
    key_financial_metrics: list[ExtractedMetric]
    portfolio_companies: list[PortfolioCompany]
    raw_extraction_available: bool
    source_document: str = "KhazanahAnnualReview2026.pdf"

# --- SYSTEM MONITORING ---
class HealthResponse(BaseModel):
    """Check the status of the vector database and the currently active model stack."""
    status: str
    chunk_count: int
    collection_name: str
    embedding_model: str
    llm_model: str
    pdf_ingested: bool
    message: str
