from fastapi import APIRouter, HTTPException
from backend.services.retriever import load_extracted_data
from backend.models.schemas import ExtractedData, ExtractedMetric, PortfolioCompany

router = APIRouter()


@router.get("/extract", response_model=ExtractedData, tags=["Extraction"])
async def get_extracted_data():
    """
    Return pre-extracted structured data from the Annual Review.
    Includes key financial metrics and portfolio companies.
    This is a static endpoint — data was extracted during ingestion.
    """
    try:
        data = load_extracted_data()

        metrics = [
            ExtractedMetric(**m)
            for m in data.get("key_financial_metrics", [])
        ]

        companies = [
            PortfolioCompany(**c)
            for c in data.get("portfolio_companies", [])
        ]

        return ExtractedData(
            key_financial_metrics=metrics,
            portfolio_companies=companies,
            raw_extraction_available=True,
            source_document=data.get("source_document", "KhazanahAnnualReview2026.pdf")
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )