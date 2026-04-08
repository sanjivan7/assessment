"""
Ingestion pipeline for Khazanah Annual Review 2026.

Two-pass extraction:
  Pass 1 — PyMuPDF: extract native text layer per page
  Pass 2 — GPT-4o-mini vision: extract structured data from page images
            (handles charts, tables, infographics invisible to text extraction)

Each page = one chunk. Chunks stored in ChromaDB with full metadata.
Extracted structured data saved as JSON for the /extract endpoint.
"""

import sys
import os
import json
import base64
import time
import pickle
from pathlib import Path

# Add project root to path so we can import backend modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz  # PyMuPDF
import numpy as np
from openai import OpenAI
import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pdf_page_to_base64(page: fitz.Page, dpi: int = 150) -> str:
    """Render a PDF page to a base64-encoded PNG string."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_bytes = pix.tobytes("png")
    return base64.standard_b64encode(img_bytes).decode("utf-8")


def classify_slide_type(native_text: str, page_num: int) -> str:
    """
    Rough heuristic classification based on native text content.
    Vision model will refine this during extraction.
    """
    text_lower = native_text.lower()
    if any(k in text_lower for k in ["rm b", "rm ", "%", "revenue", "profit", "dividend", "debt", "nav", "rav", "twrr"]):
        return "financial"
    if any(k in text_lower for k in ["portfolio", "investment", "asset class", "public market", "private market"]):
        return "portfolio"
    if any(k in text_lower for k in ["thank you", "moving forward", "2025 in review"]):
        return "section_divider"
    return "narrative"


def get_vision_extraction_prompt(slide_type: str, page_num: int) -> str:
    """
    Return a slide-type-specific prompt for GPT-4o-mini vision.
    Specific prompts outperform generic ones for financial data.
    """
    base = f"""You are extracting data from slide {page_num} of Khazanah Nasional Berhad's Annual Review 2026.
Extract ALL visible information. Return ONLY valid JSON, no markdown, no explanation.

"""
    if slide_type == "financial":
        return base + """Focus on ALL numerical values, percentages, and financial metrics.
Return JSON with this exact structure:
{
  "slide_title": "...",
  "slide_type": "financial",
  "narrative_text": "any non-numerical text",
  "key_metrics": [
    {"metric": "metric name", "value": "exact value", "unit": "RM b / % / x etc", "year": "year if shown"}
  ],
  "table_data": [
    {"headers": ["col1", "col2"], "rows": [["val1", "val2"]]}
  ],
  "chart_description": "describe any charts — trend direction, labeled values, axis labels",
  "source_note": "any source attribution shown"
}"""

    if slide_type == "portfolio":
        return base + """Focus on portfolio companies, sectors, ownership stakes, and performance data.
Return JSON with this exact structure:
{
  "slide_title": "...",
  "slide_type": "portfolio",
  "narrative_text": "...",
  "companies": [
    {"name": "company name", "sector": "sector", "ownership": "stake %", "notes": "any other detail"}
  ],
  "key_metrics": [
    {"metric": "...", "value": "...", "unit": "...", "year": "..."}
  ],
  "table_data": [],
  "chart_description": "..."
}"""

    # default / narrative
    return base + """Extract all text preserving hierarchy (titles, bullets, captions).
Return JSON with this exact structure:
{
  "slide_title": "...",
  "slide_type": "narrative",
  "narrative_text": "full text content preserving structure",
  "key_metrics": [],
  "table_data": [],
  "companies": [],
  "chart_description": "describe any visual elements"
}"""


# ---------------------------------------------------------------------------
# Pass 1 — Native text extraction
# ---------------------------------------------------------------------------

def extract_native_text(pdf_path: Path) -> list[dict]:
    """Extract text layer from each page using PyMuPDF."""
    print(f"\n[Pass 1] Extracting native text from {pdf_path.name}...")
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        pages.append({
            "page_number": page_num + 1,  # 1-indexed
            "native_text": text,
            "char_count": len(text),
        })
        print(f"  Page {page_num + 1}: {len(text)} chars extracted")
    doc.close()
    print(f"[Pass 1] Complete. {len(pages)} pages processed.")
    return pages


# ---------------------------------------------------------------------------
# Pass 2 — Vision extraction via GPT-4o-mini
# ---------------------------------------------------------------------------

def extract_vision_data(pdf_path: Path, native_pages: list[dict], client: OpenAI) -> list[dict]:
    """
    Send each page as an image to GPT-4o-mini for structured extraction.
    Handles charts, tables, and content invisible to text extraction.
    """
    print(f"\n[Pass 2] Vision extraction via GPT-4o-mini...")
    doc = fitz.open(str(pdf_path))
    enriched_pages = []

    for i, page_data in enumerate(native_pages):
        page_num = page_data["page_number"]
        print(f"  Processing page {page_num}/{len(native_pages)}...", end=" ")

        # Render page to base64 image
        page = doc[i]
        img_b64 = pdf_page_to_base64(page, dpi=150)

        # Classify slide type from native text (heuristic)
        slide_type = classify_slide_type(page_data["native_text"], page_num)
        prompt = get_vision_extraction_prompt(slide_type, page_num)

        # Call GPT-4o-mini with vision
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ]
            )
            raw_json_str = response.choices[0].message.content.strip()

            # Strip markdown code fences if model adds them
            if raw_json_str.startswith("```"):
                raw_json_str = raw_json_str.split("```")[1]
                if raw_json_str.startswith("json"):
                    raw_json_str = raw_json_str[4:]
                raw_json_str = raw_json_str.strip()

            vision_data = json.loads(raw_json_str)
            extraction_method = "vision+text" if page_data["char_count"] > 50 else "vision_only"
            print(f"OK ({slide_type})")

        except json.JSONDecodeError as e:
            print(f"JSON parse error — using native text fallback")
            vision_data = {
                "slide_title": f"Slide {page_num}",
                "slide_type": slide_type,
                "narrative_text": page_data["native_text"],
                "key_metrics": [],
                "table_data": [],
                "companies": [],
                "chart_description": "Vision extraction JSON parse failed"
            }
            extraction_method = "text_only"

        except Exception as e:
            print(f"API error: {e} — using native text fallback")
            vision_data = {
                "slide_title": f"Slide {page_num}",
                "slide_type": slide_type,
                "narrative_text": page_data["native_text"],
                "key_metrics": [],
                "table_data": [],
                "companies": [],
                "chart_description": f"Vision extraction failed: {str(e)}"
            }
            extraction_method = "text_only"

        # Merge native text + vision data
        enriched_pages.append({
            "page_number": page_num,
            "native_text": page_data["native_text"],
            "vision_data": vision_data,
            "slide_title": vision_data.get("slide_title", f"Slide {page_num}"),
            "slide_type": vision_data.get("slide_type", slide_type),
            "extraction_method": extraction_method,
            "chunk_id": f"slide_{page_num:03d}",
        })

        # Respect rate limits — 500ms between calls
        time.sleep(0.5)

    doc.close()
    print(f"[Pass 2] Complete. {len(enriched_pages)} pages enriched.")
    return enriched_pages


# ---------------------------------------------------------------------------
# Build chunk text for embedding
# ---------------------------------------------------------------------------

def build_chunk_text(page: dict) -> str:
    """
    Combine native text + vision extraction into a single
    rich text string for embedding. More context = better retrieval.
    """
    parts = []
    vd = page["vision_data"]

    # Slide title
    title = page.get("slide_title", "")
    if title:
        parts.append(f"SLIDE TITLE: {title}")

    # Native text (if meaningful)
    native = page.get("native_text", "").strip()
    if native and len(native) > 30:
        parts.append(f"TEXT: {native}")

    # Narrative from vision
    narrative = vd.get("narrative_text", "").strip()
    if narrative and narrative != native:
        parts.append(f"CONTENT: {narrative}")

    # Key metrics — most important for financial queries
    metrics = vd.get("key_metrics", [])
    if metrics:
        metric_lines = []
        for m in metrics:
            line = f"{m.get('metric', '')} {m.get('value', '')} {m.get('unit', '')} {m.get('year', '')}".strip()
            if line:
                metric_lines.append(line)
        if metric_lines:
            parts.append("METRICS: " + " | ".join(metric_lines))

    # Chart description
    chart_desc = vd.get("chart_description", "").strip()
    if chart_desc and chart_desc not in ["", "Vision extraction JSON parse failed"]:
        parts.append(f"CHART: {chart_desc}")

    # Companies
    companies = vd.get("companies", [])
    if companies:
        company_names = [c.get("name", "") for c in companies if c.get("name")]
        if company_names:
            parts.append("COMPANIES: " + ", ".join(company_names))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[dict], client: OpenAI) -> list[dict]:
    """Generate embeddings for all chunks using text-embedding-3-small."""
    print(f"\n[Embedding] Generating embeddings for {len(chunks)} chunks...")

    texts = [build_chunk_text(c) for c in chunks]

    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts
    )

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = response.data[i].embedding
        chunk["chunk_text"] = texts[i]

    print(f"[Embedding] Complete. Dimension: {len(chunks[0]['embedding'])}")
    return chunks


# ---------------------------------------------------------------------------
# Store in ChromaDB
# ---------------------------------------------------------------------------

def store_in_chromadb(chunks: list[dict]) -> chromadb.Collection:
    """Store chunks, embeddings, and metadata in ChromaDB."""
    print(f"\n[ChromaDB] Storing {len(chunks)} chunks...")

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(
        path=str(settings.chroma_dir)
    )

    # Delete existing collection to allow re-ingestion
    try:
        chroma_client.delete_collection(settings.collection_name)
        print(f"  Deleted existing collection '{settings.collection_name}'")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Prepare data for ChromaDB
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        embeddings.append(chunk["embedding"])
        documents.append(chunk["chunk_text"])
        metadatas.append({
            "page_number": chunk["page_number"],
            "slide_title": chunk["slide_title"],
            "slide_type": chunk["slide_type"],
            "extraction_method": chunk["extraction_method"],
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"[ChromaDB] Stored {collection.count()} chunks in '{settings.collection_name}'")
    return collection


# ---------------------------------------------------------------------------
# Save extracted structured data for /extract endpoint
# ---------------------------------------------------------------------------

def save_extracted_data(chunks: list[dict]) -> None:
    """
    Parse vision extraction results and save structured JSON
    for portfolio companies and key financial metrics.
    """
    print(f"\n[Structured Data] Extracting structured data...")

    all_metrics = []
    all_companies = []

    for chunk in chunks:
        vd = chunk["vision_data"]
        page_num = chunk["page_number"]

        # Financial metrics
        for m in vd.get("key_metrics", []):
            if m.get("value") and m.get("metric"):
                all_metrics.append({
                    "metric_name": m.get("metric", ""),
                    "value": m.get("value", ""),
                    "unit": m.get("unit", ""),
                    "year": m.get("year", ""),
                    "context": f"Slide {page_num}: {chunk['slide_title']}",
                    "page_number": page_num
                })

        # Portfolio companies
        for c in vd.get("companies", []):
            if c.get("name"):
                all_companies.append({
                    "name": c.get("name", ""),
                    "sector": c.get("sector", ""),
                    "ownership_stake": c.get("ownership", ""),
                    "notes": c.get("notes", ""),
                    "page_number": page_num
                })

    structured = {
        "key_financial_metrics": all_metrics,
        "portfolio_companies": all_companies,
        "total_slides_processed": len(chunks),
        "source_document": settings.pdf_filename
    }

    settings.extracted_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.extracted_dir / "structured_data.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)

    print(f"[Structured Data] Saved {len(all_metrics)} metrics, {len(all_companies)} companies")
    print(f"  → {output_path}")


# ---------------------------------------------------------------------------
# Save full chunks for full-context mode
# ---------------------------------------------------------------------------

def save_full_context(chunks: list[dict]) -> None:
    """Save all chunk texts concatenated for full-context RAG mode."""
    full_text_parts = []
    for chunk in chunks:
        full_text_parts.append(
            f"=== SLIDE {chunk['page_number']}: {chunk['slide_title']} ===\n{chunk['chunk_text']}"
        )

    full_context = "\n\n".join(full_text_parts)

    output_path = settings.extracted_dir / "full_context.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_context)

    print(f"[Full Context] Saved full context ({len(full_context)} chars) → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Khazanah Annual Review 2026 — Ingestion Pipeline")
    print("=" * 60)

    # Validate PDF exists
    pdf_path = settings.docs_dir / settings.pdf_filename
    if not pdf_path.exists():
        print(f"\nERROR: PDF not found at {pdf_path}")
        print(f"Place the PDF in the docs/ folder and check the filename in .env")
        sys.exit(1)

    print(f"\nPDF found: {pdf_path}")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"LLM model: {settings.llm_model}")

    # Initialise OpenAI client
    client = OpenAI(api_key=settings.openai_api_key)

    # Pass 1 — native text
    native_pages = extract_native_text(pdf_path)

    # Pass 2 — vision extraction
    enriched_chunks = extract_vision_data(pdf_path, native_pages, client)

    # Save raw extraction for debugging
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = settings.raw_dir / "raw_extraction.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        # Don't save embeddings in raw dump — too large
        safe_chunks = [{k: v for k, v in c.items() if k != "embedding"} for c in enriched_chunks]
        json.dump(safe_chunks, f, indent=2, ensure_ascii=False)
    print(f"\n[Raw] Saved raw extraction → {raw_path}")

    # Embed chunks
    enriched_chunks = embed_chunks(enriched_chunks, client)

    # Store in ChromaDB
    store_in_chromadb(enriched_chunks)

    # Save structured data for /extract endpoint
    save_extracted_data(enriched_chunks)

    # Save full context for full-context mode
    save_full_context(enriched_chunks)

    print("\n" + "=" * 60)
    print("Ingestion complete. You can now start the FastAPI backend.")
    print("=" * 60)


if __name__ == "__main__":
    main()