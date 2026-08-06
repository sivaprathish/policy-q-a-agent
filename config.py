from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Embedding configuration
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

EMBEDDING_DEVICE = os.getenv(
    "EMBEDDING_DEVICE",
    "cpu",
)

EMBEDDING_BATCH_SIZE = int(
    os.getenv(
        "EMBEDDING_BATCH_SIZE",
        "32",
    )
)


# ============================================================
# FAISS index paths
# ============================================================

DOCUMENT_FAISS_INDEX = (
    VECTOR_DB_DIR / "documents.faiss"
)

DOCUMENT_METADATA_FILE = (
    VECTOR_DB_DIR / "documents_metadata.json"
)

SECTION_FAISS_INDEX = (
    VECTOR_DB_DIR / "sections.faiss"
)

SECTION_METADATA_FILE = (
    VECTOR_DB_DIR / "sections_metadata.json"
)

CHUNK_FAISS_INDEX = (
    VECTOR_DB_DIR / "chunks.faiss"
)

CHUNK_METADATA_FILE = (
    VECTOR_DB_DIR / "chunks_metadata.json"
)


# ============================================================
# LLM configuration
# ============================================================

LLM_API_KEY = os.getenv(
    "LLM_API_KEY",
    "",
)

LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "meta/llama-3.1-70b-instruct",
)

LLM_TEMPERATURE = float(
    os.getenv(
        "LLM_TEMPERATURE",
        "0.1",
    )
)

LLM_MAX_TOKENS = int(
    os.getenv(
        "LLM_MAX_TOKENS",
        "900",
    )
)


# ============================================================
# Hierarchical retrieval configuration
# ============================================================

DOCUMENT_TOP_K = int(
    os.getenv(
        "DOCUMENT_TOP_K",
        "2",
    )
)

SECTION_TOP_K = int(
    os.getenv(
        "SECTION_TOP_K",
        "5",
    )
)

CHUNK_TOP_K = int(
    os.getenv(
        "CHUNK_TOP_K",
        "6",
    )
)


# ============================================================
# Retrieval thresholds
# ============================================================

DOCUMENT_SCORE_THRESHOLD = float(
    os.getenv(
        "DOCUMENT_SCORE_THRESHOLD",
        "0.15",
    )
)

SECTION_SCORE_THRESHOLD = float(
    os.getenv(
        "SECTION_SCORE_THRESHOLD",
        "0.20",
    )
)

CHUNK_SCORE_THRESHOLD = float(
    os.getenv(
        "CHUNK_SCORE_THRESHOLD",
        "0.25",
    )
)