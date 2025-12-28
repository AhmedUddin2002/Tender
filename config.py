from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", DATA_DIR / "raw"))
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", DATA_DIR / "processed"))
ARCHIVE_DATA_DIR = Path(os.getenv("ARCHIVE_DATA_DIR", DATA_DIR / "archive"))

LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1/chat/completions")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "google/gemma-3-1b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L12-v2")

AI_CACHE_PATH = Path(os.getenv("AI_CACHE_PATH", PROCESSED_DATA_DIR / "ai_cache.json"))
VECTOR_DB_PATH = Path(os.getenv("VECTOR_DB_PATH", PROCESSED_DATA_DIR / "vector_store"))

# Ensure directories exist at import time for smoother DX
for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, ARCHIVE_DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def timestamped_filename(name: str, suffix: str = "csv", prefix: Optional[str] = None) -> Path:
    """Generate a timestamped filename within the processed directory."""
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    parts = [p for p in [prefix, name, stamp] if p]
    filename = "_".join(parts)
    if suffix:
        filename = f"{filename}.{suffix.lstrip('.')}"
    return PROCESSED_DATA_DIR / filename


LATEST_COMBINED_PATH = PROCESSED_DATA_DIR / "tenders_combined_latest.csv"
LATEST_CLEAN_PATH = PROCESSED_DATA_DIR / "tenders_clean_latest.csv"

# Raw artefacts naming helpers

def raw_export_path(source: str, extension: str = "csv") -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"{source}_{stamp}.{extension.lstrip('.')}"
    return RAW_DATA_DIR / filename
