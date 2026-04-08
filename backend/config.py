from pydantic_settings import BaseSettings
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # API Keys
    openai_api_key: str
    gemini_api_key: str = ""  # optional fallback

    # Paths
    docs_dir: Path = BASE_DIR / "docs"
    chroma_dir: Path = BASE_DIR / "backend" / "data" / "chroma_db"
    extracted_dir: Path = BASE_DIR / "backend" / "data" / "extracted"
    raw_dir: Path = BASE_DIR / "backend" / "data" / "raw"

    # PDF filename
    pdf_filename: str = "KhazanahAnnualReview2026.pdf"

    # Embedding model
    embedding_model: str = "text-embedding-3-small"

    # LLM model
    llm_model: str = "gpt-4o-mini"

    # RAG settings
    top_k: int = 5
    similarity_threshold: float = 0.30

    # ChromaDB collection name
    collection_name: str = "khazanah_kar_2026"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()