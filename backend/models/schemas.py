from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NO_CONTEXT = "no_context"


class QueryRequest(BaseModel):
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
    chunk_id: str
    page_number: int
    slide_title: str
    slide_type: str
    extraction_method: str
    relevance_score: float
    text_preview: str  # first 200 chars of chunk text


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    confidence: ConfidenceLevel
    retrieval_scores: list[float]
    full_context_mode_used: bool = False
    query: str


class ExtractedMetric(BaseModel):
    metric_name: str
    value: str
    unit: Optional[str] = None
    year: Optional[str] = None
    context: Optional[str] = None


class PortfolioCompany(BaseModel):
    name: str
    sector: Optional[str] = None
    ownership_stake: Optional[str] = None
    notes: Optional[str] = None


class ExtractedData(BaseModel):
    key_financial_metrics: list[ExtractedMetric]
    portfolio_companies: list[PortfolioCompany]
    raw_extraction_available: bool
    source_document: str = "KhazanahAnnualReview2026.pdf"


class HealthResponse(BaseModel):
    status: str
    chunk_count: int
    collection_name: str
    embedding_model: str
    llm_model: str
    pdf_ingested: bool
    message: str